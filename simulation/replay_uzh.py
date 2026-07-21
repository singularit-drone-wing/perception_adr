import socket
import struct
import time
import os
import sys
import numpy as np

# ==========================================
# 1. Packet Format Configurations
# ==========================================

# IMU Packet: 'I' (char), timestamp (double), ax, ay, az, gx, gy, gz (6x double)
IMU_FORMAT = "<c d d d d d d d"
# Camera Header: 'C' (char), timestamp (double), image_size (uint32)
CAM_HEADER_FORMAT = "<c d I"
# State Packet: 'S' (char), timestamp (double), px, py, pz, vx, vy, vz, qw, qx, qy, qz, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z (16x double)
STATE_FORMAT = "<c d d d d d d d d d d d d d d d d d"

# Ports
CPP_RX_PORT = 12345
CPP_TX_PORT = 12346

def load_groundtruth(gt_path):
    """Loads ground truth timestamps, position vectors, and quaternions."""
    gt_data = []
    if not os.path.exists(gt_path):
        return None
    with open(gt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qw, qx, qy, qz = float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])
                gt_data.append((ts, np.array([tx, ty, tz]), np.array([qw, qx, qy, qz])))
    return gt_data

def find_closest_groundtruth(gt_data, timestamp):
    """Finds nearest ground truth pose for a given timestamp."""
    if not gt_data:
        return None, None
    timestamps = [g[0] for g in gt_data]
    idx = np.searchsorted(timestamps, timestamp)
    if idx == 0:
        return gt_data[0][1], gt_data[0][2]
    if idx >= len(gt_data):
        return gt_data[-1][1], gt_data[-1][2]
    if abs(timestamps[idx-1] - timestamp) < abs(timestamps[idx] - timestamp):
        return gt_data[idx-1][1], gt_data[idx-1][2]
    return gt_data[idx][1], gt_data[idx][2]

def umeyama_alignment(x_est, x_gt):
    """
    Computes optimal 6DoF SE(3) rotation (R) and translation (t) alignment
    between estimated positions (x_est) and ground truth positions (x_gt)
    using Umeyama's algorithm (Standard VIO / SLAM benchmark tool format).
    """
    if len(x_est) < 3:
        return x_est, np.eye(3), np.zeros(3)
    
    mu_est = np.mean(x_est, axis=0)
    mu_gt = np.mean(x_gt, axis=0)

    p_est = x_est - mu_est
    p_gt = x_gt - mu_gt

    H = p_est.T @ p_gt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Special reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_gt - R @ mu_est
    aligned_est = (R @ x_est.T).T + t
    return aligned_est, R, t

def main():
    print("=========================================================")
    print("UZH-FPV VIO Benchmark Evaluation (SE3 Trajectory Alignment)")
    print("=========================================================")

    dataset_dir = "datasets/uzh-fpv-indoor-forward-davis3"
    imu_file = os.path.join(dataset_dir, "imu.txt")
    images_file = os.path.join(dataset_dir, "images.txt")
    gt_file = os.path.join(dataset_dir, "groundtruth.txt")

    if not os.path.exists(imu_file) or not os.path.exists(images_file):
        print(f"Error: Dataset files not found in {dataset_dir}")
        print("Please ensure the UZH dataset is extracted at datasets/uzh-fpv-indoor-forward-davis3/")
        sys.exit(1)

    # 1. Load Ground Truth data for benchmark comparison
    gt_data = load_groundtruth(gt_file)
    if gt_data:
        print(f"[Init] Loaded {len(gt_data)} Ground Truth poses from groundtruth.txt")
    else:
        print("[Init] Warning: groundtruth.txt not found, proceeding without GT benchmark.")

    # 2. Build Event Stream (IMU + Camera frames sorted chronologically)
    events = []

    # Read IMU data
    print("[Init] Parsing imu.txt...")
    with open(imu_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                ts = float(parts[1])
                gx, gy, gz = float(parts[2]), float(parts[3]), float(parts[4])
                ax, ay, az = float(parts[5]), float(parts[6]), float(parts[7])
                events.append((ts, 'IMU', (ax, ay, az, gx, gy, gz)))

    # Read Camera frames list
    print("[Init] Parsing images.txt...")
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                ts = float(parts[1])
                rel_img_path = parts[2]
                full_img_path = os.path.join(dataset_dir, rel_img_path)
                events.append((ts, 'CAM', full_img_path))

    # Sort events by timestamp
    events.sort(key=lambda x: x[0])
    print(f"[Init] Total timeline events queued: {len(events)} (IMU + Camera)")

    # 3. Socket Configuration
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind(("127.0.0.1", CPP_TX_PORT))
    rx_sock.setblocking(False)
    print(f"[Init] UDP Sockets initialized (TX Port: {CPP_RX_PORT}, RX Port: {CPP_TX_PORT})")

    # Replay Parameters
    replay_speed = 0.0  # Default to max speed for benchmark evaluation unless specified
    if len(sys.argv) > 1:
        try:
            replay_speed = float(sys.argv[1])
        except ValueError:
            pass

    speed_str = "Max Throughput Benchmark" if replay_speed <= 0 else f"{replay_speed}x Realtime"
    print(f"[Init] Replay Mode: {speed_str}")

    start_sim_time = events[0][0]
    start_wall_time = time.time()

    est_timestamps = []
    est_positions = []
    gt_positions = []

    print("[Replay] Executing dataset replay & state trajectory capture...")

    for event in events:
        ts, event_type, data = event

        # Maintain pace if replay_speed > 0
        if replay_speed > 0:
            elapsed_sim = ts - start_sim_time
            target_wall_time = start_wall_time + (elapsed_sim / replay_speed)
            now = time.time()
            if target_wall_time > now:
                time.sleep(target_wall_time - now)

        # Dispatch event to C++ node
        if event_type == 'IMU':
            ax, ay, az, gx, gy, gz = data
            imu_packet = struct.pack(
                IMU_FORMAT, b'I', ts,
                ax, ay, az, gx, gy, gz
            )
            tx_sock.sendto(imu_packet, ("127.0.0.1", CPP_RX_PORT))

        elif event_type == 'CAM':
            img_path = data
            if os.path.exists(img_path):
                with open(img_path, 'rb') as img_f:
                    jpeg_bytes = img_f.read()
                cam_header = struct.pack(CAM_HEADER_FORMAT, b'C', ts, len(jpeg_bytes))
                tx_sock.sendto(cam_header + jpeg_bytes, ("127.0.0.1", CPP_RX_PORT))

        # Non-blocking check for EKF state feedback from C++ node
        try:
            while True:
                recv_data, _ = rx_sock.recvfrom(2048)
                if len(recv_data) == struct.calcsize(STATE_FORMAT):
                    fields = struct.unpack(STATE_FORMAT, recv_data)
                    ts_est = fields[1]
                    p_est = np.array([fields[2], fields[3], fields[4]])

                    gt_pos, _ = find_closest_groundtruth(gt_data, ts_est)
                    if gt_pos is not None:
                        est_timestamps.append(ts_est)
                        est_positions.append(p_est)
                        gt_positions.append(gt_pos)
        except BlockingIOError:
            pass

    tx_sock.close()
    rx_sock.close()

    # Convert to NumPy arrays
    est_pos_arr = np.array(est_positions)
    gt_pos_arr = np.array(gt_positions)

    print("\n=========================================================")
    print("               VIO BENCHMARK RESULTS                     ")
    print("=========================================================")
    print(f"Total Evaluated Trajectory Samples : {len(est_pos_arr)}")

    if len(est_pos_arr) > 0:
        # 1. Initial Frame Alignment (t = 0 alignment)
        init_offset = gt_pos_arr[0] - est_pos_arr[0]
        pos_init_aligned = est_pos_arr + init_offset
        errors_init_aligned = np.linalg.norm(pos_init_aligned - gt_pos_arr, axis=1)

        # 2. Full SE(3) Umeyama Trajectory Alignment (Standard evo / ATE benchmark)
        pos_se3_aligned, R_opt, t_opt = umeyama_alignment(est_pos_arr, gt_pos_arr)
        ate_errors = np.linalg.norm(pos_se3_aligned - gt_pos_arr, axis=1)

        print("\n--- A. Origin Alignment (t = 0 Alignment) ---")
        print(f"Initial Position Offset    : [{init_offset[0]:.4f}, {init_offset[1]:.4f}, {init_offset[2]:.4f}] m")
        print(f"t=0 Aligned ATE Mean Error : {np.mean(errors_init_aligned):.4f} m")
        print(f"t=0 Aligned ATE RMSE       : {np.sqrt(np.mean(np.square(errors_init_aligned))):.4f} m")

        print("\n--- B. SE(3) Umeyama Trajectory Alignment (Standard VIO ATE Benchmark) ---")
        print(f"Absolute Trajectory Error (ATE) RMSE : {np.sqrt(np.mean(np.square(ate_errors))):.4f} m")
        print(f"ATE Mean Error                       : {np.mean(ate_errors):.4f} m")
        print(f"ATE Median Error                     : {np.median(ate_errors):.4f} m")
        print(f"ATE Min / Max Error                  : {np.min(ate_errors):.4f} m / {np.max(ate_errors):.4f} m")
        print(f"ATE Standard Deviation               : {np.std(ate_errors):.4f} m")
        print("=========================================================")
    else:
        print("Error: No estimated trajectory states were received from the C++ node.")

if __name__ == "__main__":
    main()
