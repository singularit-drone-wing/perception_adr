import socket
import struct
import time
import math
import numpy as np
import os
import sys

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

def main():
    print("=========================================")
    # Print absolute path to simulate_drone.py using python file link formatting
    print("Autonomous Drone Racing Python Simulator Client")
    print("=========================================")

    # TODO: Render synthetic gate projections using Kannala-Brandt fisheye model
    # to provide meaningful visual measurements. Currently, static UZH images are
    # skipped because their gate positions don't match the C++ gate map, causing
    # the EKF to diverge. Gate rendering should project 1.5m x 1.5m gate corners
    # onto a blank canvas using the DAVIS240C calibration (K, D) and send as JPEG.
    # See simulation/generate_synthetic_gates.py for the projection approach.

    # Set up UDP sockets
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind(("127.0.0.1", CPP_TX_PORT))
    rx_sock.setblocking(False)

    print(f"[Init] Sockets configured. TX Port: {CPP_RX_PORT}, RX Port: {CPP_TX_PORT}")
    
    # Simulation trajectory parameters
    t_start = time.time()
    t_sim = 0.0
    duration = 5.0  # run simulation for 5 seconds
    imu_rate = 200.0  # 200 Hz IMU packet rate
    cam_rate = 10.0   # 10 Hz camera frame rate

    dt_imu = 1.0 / imu_rate
    dt_cam = 1.0 / cam_rate

    last_imu_time = 0.0
    last_cam_time = 0.0

    # True biases (for ground truth reference)
    true_acc_bias = np.array([0.01, -0.01, 0.02])
    true_gyro_bias = np.array([0.001, -0.001, 0.002])

    print("[Sim] Starting closed-loop estimation loop...")

    # Statistics accumulators
    states_received = 0
    start_loop = time.time()

    while t_sim < duration:
        t_now = time.time() - start_loop

        # 1. Simulate High-Rate IMU (200Hz)
        if t_now - last_imu_time >= dt_imu:
            t_sim = t_now
            last_imu_time = t_now

            # Drone flying in a circle around center [0, 5, 5]
            radius = 3.0
            omega = 1.0 # 1 rad/s
            
            # Position: p_WB(t)
            px = radius * math.cos(omega * t_sim)
            py = radius * math.sin(omega * t_sim) + 5.0
            pz = 5.0 + 1.0 * math.sin(0.5 * omega * t_sim)

            # Compute body-frame acceleration using R_WB^T * (a_kin_world - g_world)
            # The drone yaws at rate omega around z, so the centripetal acceleration
            # is always in body -x, and y-component cancels due to matching yaw rate.
            # a_kin_world = [-R*omega^2*cos(omega*t), -R*omega^2*sin(omega*t), -0.25*omega^2*sin(0.5*omega*t)]
            # R_WB^T rotates a_kin_world - g_world by -omega*t around z, giving:
            #
            #   a_body_x = -R*omega^2                          (constant centripetal)
            #   a_body_y = 0                                    (coriolis cancellation)
            #   a_body_z = -0.25*omega^2*sin(0.5*omega*t) - 9.81
            #
            a_body_x = -radius * (omega**2)
            a_body_y = 0.0
            a_body_z = -1.0 * (0.25 * omega**2) * math.sin(0.5 * omega * t_sim) - 9.81

            ax_meas = a_body_x + true_acc_bias[0] + np.random.normal(0, 0.02)
            ay_meas = a_body_y + true_acc_bias[1] + np.random.normal(0, 0.02)
            az_meas = a_body_z + true_acc_bias[2] + np.random.normal(0, 0.02)

            # Gyro measures: angular rates + bias + noise
            # Simple yaw rotation rate (omega = 1.0 rad/s)
            gx_meas = 0.0 + true_gyro_bias[0] + np.random.normal(0, 0.005)
            gy_meas = 0.0 + true_gyro_bias[1] + np.random.normal(0, 0.005)
            gz_meas = omega + true_gyro_bias[2] + np.random.normal(0, 0.005)

            # Pack and send IMU packet
            imu_packet = struct.pack(
                IMU_FORMAT, b'I', t_sim,
                ax_meas, ay_meas, az_meas,
                gx_meas, gy_meas, gz_meas
            )
            tx_sock.sendto(imu_packet, ("127.0.0.1", CPP_RX_PORT))

        # 2. Simulate Camera Frames (10Hz) — DISABLED: see note above about synthetic gates
        #if t_now - last_cam_time >= dt_cam:
        #    last_cam_time = t_now
        #    cam_header = struct.pack(CAM_HEADER_FORMAT, b'C', t_sim, len(jpeg_bytes))
        #    tx_sock.sendto(cam_header + jpeg_bytes, ("127.0.0.1", CPP_RX_PORT))

        # 3. Check for EKF state feedback from C++ node
        try:
            data, _ = rx_sock.recvfrom(2048)
            if len(data) == struct.calcsize(STATE_FORMAT):
                fields = struct.unpack(STATE_FORMAT, data)
                ts = fields[1]
                px_est, py_est, pz_est = fields[2], fields[3], fields[4]
                vx_est, vy_est, vz_est = fields[5], fields[6], fields[7]
                qw, qx, qy, qz = fields[8], fields[9], fields[10], fields[11]
                ba_x, ba_y, ba_z = fields[12], fields[13], fields[14]
                bg_x, bg_y, bg_z = fields[15], fields[16], fields[17]

                states_received += 1
                if states_received % 5 == 0:
                    print(f"[Feedback] t={ts:.3f}s | EKF Global State Vector:")
                    print(f"  Position (p_WB)  : [{px_est:8.4f}, {py_est:8.4f}, {pz_est:8.4f}] m")
                    print(f"  Velocity (v_W)   : [{vx_est:8.4f}, {vy_est:8.4f}, {vz_est:8.4f}] m/s")
                    print(f"  Attitude (q_WB)  : [{qw:8.5f}, {qx:8.5f}, {qy:8.5f}, {qz:8.5f}] (w,x,y,z)")
                    print(f"  Acc Bias (b_a)   : [{ba_x:8.5f}, {ba_y:8.5f}, {ba_z:8.5f}] m/s²")
                    print(f"  Gyro Bias (b_g)  : [{bg_x:8.5f}, {bg_y:8.5f}, {bg_z:8.5f}] rad/s")
                    print("-" * 60)
        except BlockingIOError:
            pass

        # Sleep a little to prevent high CPU utilization
        time.sleep(0.001)

    print("=========================================")
    print("Simulation Loop Finished.")
    print(f"Total State Packets Received from EKF: {states_received}")
    print("=========================================")

    tx_sock.close()
    rx_sock.close()

if __name__ == "__main__":
    main()
