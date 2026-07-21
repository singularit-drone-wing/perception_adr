#include "pipeline_utils.h"
#include "vision_pipeline.h"
#include "ekf.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/highgui.hpp>

// start perception_node:
// ./cpp_pipeline/build/perception_node ./weights/best.onnx 127.0.0.1

// ==========================================
// 1. Packet Definitions
// ==========================================

struct IMUPacket {
    char type; // 'I'
    double timestamp;
    double ax, ay, az;
    double gx, gy, gz;
} __attribute__((packed));

struct CameraPacketHeader {
    char type; // 'C'
    double timestamp;
    uint32_t size;
} __attribute__((packed));

struct StatePacket {
    char type; // 'S'
    double timestamp;
    double px, py, pz;
    double vx, vy, vz;
    double qw, qx, qy, qz;
    double ba_x, ba_y, ba_z;
    double bg_x, bg_y, bg_z;
} __attribute__((packed));

// Struct to pass camera frames between thread 2 and thread 1
struct ImageFrame {
    double timestamp;
    std::vector<uchar> jpeg_data;
};

// ==========================================
// 2. Global Shared Objects & Configuration
// ==========================================

std::atomic<bool> running(true);

// Queues for inter-thread communication
ThreadSafeQueue<ImageFrame> camera_queue;
ThreadSafeQueue<PoseMeasurement> measurement_queue;

// EKF state and prediction ring buffer
EKF ekf;
RingBuffer ring_buffer(2000); // 2 seconds of history at 1KHz

// UDP ports
const int RX_PORT = 12345; // C++ listens to Python inputs
const int TX_PORT = 12346; // Python listens to C++ EKF state

// ==========================================
// 3. Thread Functions
// ==========================================

// Thread 1: Vision Pipeline Thread (30-60 Hz)
void vision_thread_func(const std::string& model_path, 
                        const cv::Mat& K, const cv::Mat& D, 
                        const Eigen::Matrix3d& R_CB, const Eigen::Vector3d& t_CB,
                        const std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>>& gate_map) {
    std::cout << "[Thread-Vision] Initializing VisionPipeline..." << std::endl;
    std::unique_ptr<VisionPipeline> pipeline;
    try {
        pipeline = std::make_unique<VisionPipeline>(model_path, K, D, R_CB, t_CB, 0.5f, 0.45f);
        std::cout << "[Thread-Vision] VisionPipeline loaded." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Thread-Vision] Fatal Error: " << e.what() << std::endl;
        running = false;
        return;
    }

    while (running) {
        auto frame_opt = camera_queue.pop_with_timeout(std::chrono::milliseconds(100));
        if (!frame_opt) continue; // Timeout, check if running

        // Decode JPEG bytes to OpenCV matrix
        cv::Mat img = cv::imdecode(frame_opt->jpeg_data, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[Thread-Vision] Failed to decode incoming JPEG frame!" << std::endl;
            continue;
        }

        // Get the latest drone position estimate from the EKF for gate matching
        KinematicState cur_state = ekf.get_state(frame_opt->timestamp);

        // Run detection and gate matching
        PoseMeasurement measurement;
        double max_match_distance = 15.0; // Outlier rejection threshold in meters
        bool matched = pipeline->process_frame_with_gate_matching(
            img, 0.0f, 0.0f, img.cols, img.rows,
            gate_map, cur_state.position, frame_opt->timestamp, max_match_distance, measurement);

        if (matched) {
            measurement_queue.push(measurement);
        }
    }
    std::cout << "[Thread-Vision] Exited." << std::endl;
}

// Thread 3: EKF Update & State Sender Thread (Event-Driven)
void ekf_update_thread_func(const std::string& python_ip) {
    std::cout << "[Thread-EKFUpdate] Initializing socket..." << std::endl;
    int tx_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (tx_fd < 0) {
        std::cerr << "[Thread-EKFUpdate] Fatal: Failed to create TX socket!" << std::endl;
        running = false;
        return;
    }
    int opt = 1;
    setsockopt(tx_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in dest_addr{};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(TX_PORT);
    inet_pton(AF_INET, python_ip.c_str(), &dest_addr.sin_addr);

    std::cout << "[Thread-EKFUpdate] Ready to send updates to: " << python_ip << ":" << TX_PORT << std::endl;

    while (running) {
        auto meas_opt = measurement_queue.pop_with_timeout(std::chrono::milliseconds(100));
        if (!meas_opt) continue;

        // Query prediction ring buffer matching the visual capture time (t_cam)
        auto hist_state = ring_buffer.get_state_at(meas_opt->timestamp);
        if (!hist_state) {
            std::cerr << "[Thread-EKFUpdate] Delay compensation failed: no EKF history at t=" 
                      << meas_opt->timestamp << std::endl;
            continue;
        }

        // Run EKF measurement correction update
        bool success = ekf.update(*meas_opt, *hist_state);
        if (success) {
            // Get the updated state from EKF
            KinematicState corrected = ekf.get_state(meas_opt->timestamp);

            // Pack EKF state into UDP packet
            StatePacket packet;
            packet.type = 'S';
            packet.timestamp = corrected.timestamp;
            packet.px = corrected.position.x();
            packet.py = corrected.position.y();
            packet.pz = corrected.position.z();
            packet.vx = corrected.velocity.x();
            packet.vy = corrected.velocity.y();
            packet.vz = corrected.velocity.z();
            packet.qw = corrected.quaternion.w();
            packet.qx = corrected.quaternion.x();
            packet.qy = corrected.quaternion.y();
            packet.qz = corrected.quaternion.z();
            packet.ba_x = corrected.acc_bias.x();
            packet.ba_y = corrected.acc_bias.y();
            packet.ba_z = corrected.acc_bias.z();
            packet.bg_x = corrected.gyro_bias.x();
            packet.bg_y = corrected.gyro_bias.y();
            packet.bg_z = corrected.gyro_bias.z();

            sendto(tx_fd, &packet, sizeof(StatePacket), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
        }
    }

    close(tx_fd);
    std::cout << "[Thread-EKFUpdate] Exited." << std::endl;
}

// Thread 2: IMU Propagation & UDP Receiver Thread (Runs at 500-1000 Hz)
int main(int argc, char** argv) {
    std::string model_path = "../weights/best.onnx";
    std::string python_ip = "127.0.0.1";

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        python_ip = argv[2];
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "Autonomous Drone Racing C++ Perception Node" << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. Initialize calibration constants
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        172.98992850734132,   0.0,                163.33639726024606,
          0.0,                172.98303181090185, 134.99537889030861,
          0.0,                  0.0,                1.0);

    cv::Mat D = (cv::Mat_<double>(4, 1) << 
        -0.027576733308582076,
        -0.006593578674675004,
         0.0008566938165177085,
        -0.00030899587045247486);

    Eigen::Matrix3d R_CB;
    R_CB <<  0.9999711474430529,     0.0013817010649267755, -0.007469617365767657,
            -0.0014085305353606873,  0.9999925720306121,    -0.00358774655345255,
             0.007464604688444933,   0.0035981642219379494,  0.9999656658561218;

    Eigen::Vector3d t_CB(0.00018050225881571712, -0.004316353415695194, -0.027547385763471585);

    // 2. Define track gate map coordinates
    std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> gate_map = {
        { Eigen::Vector3d(0.0, 0.0, 3.0), Eigen::Quaterniond::Identity() },
        { Eigen::Vector3d(5.0, 5.0, 5.0), Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ())) },
        { Eigen::Vector3d(0.0, 10.0, 7.0), Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ())) },
        { Eigen::Vector3d(-5.0, 5.0, 5.0), Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI / 2.0, Eigen::Vector3d::UnitZ())) }
    };

    // 3. Initialize EKF initial state (matches UZH dataset starting ground truth pose)
    KinematicState initial_state;
    initial_state.timestamp = 1540820236.534;
    initial_state.position = Eigen::Vector3d(7.60526198985024, 0.240529565132054, -0.754395431415226);
    initial_state.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
    initial_state.quaternion = Eigen::Quaterniond(0.278314235606225, -0.269262241428808, -0.661934430325135, 0.641780212806374);
    initial_state.acc_bias = Eigen::Vector3d::Zero();
    initial_state.gyro_bias = Eigen::Vector3d::Zero();
    ekf.set_state(initial_state);

    // 4. Spawn threads
    std::thread vision_thread(vision_thread_func, model_path, K, D, R_CB, t_CB, gate_map);
    std::thread ekf_thread(ekf_update_thread_func, python_ip);

    // 5. Initialize RX socket on port 12345
    std::cout << "[Main] Initializing UDP RX socket on port " << RX_PORT << "..." << std::endl;
    int rx_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (rx_fd < 0) {
        std::cerr << "[Main] Fatal: Failed to create RX socket!" << std::endl;
        running = false;
    }
    int opt = 1;
    setsockopt(rx_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in rx_addr{};
    rx_addr.sin_family = AF_INET;
    rx_addr.sin_addr.s_addr = INADDR_ANY;
    rx_addr.sin_port = htons(RX_PORT);

    if (bind(rx_fd, (struct sockaddr*)&rx_addr, sizeof(rx_addr)) < 0) {
        std::cerr << "[Main] Fatal: Failed to bind RX socket!" << std::endl;
        running = false;
    }

    // Initialize TX socket for continuous telemetry output
    int main_tx_fd = socket(AF_INET, SOCK_DGRAM, 0);
    setsockopt(main_tx_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in main_dest_addr{};
    main_dest_addr.sin_family = AF_INET;
    main_dest_addr.sin_port = htons(TX_PORT);
    inet_pton(AF_INET, python_ip.c_str(), &main_dest_addr.sin_addr);

    // Set buffer size to ensure high bandwidth frames are not dropped
    int rcvbuf_size = 2 * 1024 * 1024; // 2MB
    setsockopt(rx_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));

    std::vector<uchar> buffer(65536);
    double last_imu_time = -1.0;
    uint64_t imu_count = 0;

    std::cout << "[Main] RX Listener Thread started." << std::endl;

    while (running) {
        ssize_t bytes_recvd = recv(rx_fd, buffer.data(), buffer.size(), 0);
        if (bytes_recvd <= 0) continue;

        char packet_type = buffer[0];

        if (packet_type == 'I') {
            // Process IMU packet (Thread 2)
            if (bytes_recvd < static_cast<ssize_t>(sizeof(IMUPacket))) continue;
            IMUPacket* imu = reinterpret_cast<IMUPacket*>(buffer.data());

            double dt = 0.001; // default fallback
            if (last_imu_time > 0.0) {
                dt = imu->timestamp - last_imu_time;
            }
            if (dt <= 0.0 || dt > 0.1) dt = 0.001; // clip anomaly

            last_imu_time = imu->timestamp;

            // Run high-rate EKF prediction step
            Eigen::Vector3d acc(imu->ax, imu->ay, imu->az);
            Eigen::Vector3d gyro(imu->gx, imu->gy, imu->gz);
            ekf.predict(acc, gyro, dt);

            // Log prediction in ring buffer
            KinematicState current_state = ekf.get_state(imu->timestamp);
            ring_buffer.insert(current_state);

            // Transmit state feedback telemetry every 10th IMU packet (~50 Hz)
            imu_count++;
            if (imu_count % 10 == 0) {
                StatePacket packet;
                packet.type = 'S';
                packet.timestamp = current_state.timestamp;
                packet.px = current_state.position.x();
                packet.py = current_state.position.y();
                packet.pz = current_state.position.z();
                packet.vx = current_state.velocity.x();
                packet.vy = current_state.velocity.y();
                packet.vz = current_state.velocity.z();
                packet.qw = current_state.quaternion.w();
                packet.qx = current_state.quaternion.x();
                packet.qy = current_state.quaternion.y();
                packet.qz = current_state.quaternion.z();
                packet.ba_x = current_state.acc_bias.x();
                packet.ba_y = current_state.acc_bias.y();
                packet.ba_z = current_state.acc_bias.z();
                packet.bg_x = current_state.gyro_bias.x();
                packet.bg_y = current_state.gyro_bias.y();
                packet.bg_z = current_state.gyro_bias.z();
                sendto(main_tx_fd, &packet, sizeof(StatePacket), 0, (struct sockaddr*)&main_dest_addr, sizeof(main_dest_addr));
            }

        } else if (packet_type == 'C') {
            // Process Camera packet
            if (bytes_recvd < static_cast<ssize_t>(sizeof(CameraPacketHeader))) continue;
            CameraPacketHeader* header = reinterpret_cast<CameraPacketHeader*>(buffer.data());

            if (bytes_recvd < static_cast<ssize_t>(sizeof(CameraPacketHeader) + header->size)) {
                std::cerr << "[Main] Warning: Incomplete camera packet received!" << std::endl;
                continue;
            }

            ImageFrame frame;
            frame.timestamp = header->timestamp;
            frame.jpeg_data.assign(buffer.begin() + sizeof(CameraPacketHeader), 
                                   buffer.begin() + sizeof(CameraPacketHeader) + header->size);

            // Push to Vision queue (dispatches to Thread 1 asynchronously)
            camera_queue.push(std::move(frame));
        }
    }

    // Clean up
    close(rx_fd);
    
    if (vision_thread.joinable()) vision_thread.join();
    if (ekf_thread.joinable()) ekf_thread.join();

    std::cout << "[Main] Node shut down successfully." << std::endl;
    return 0;
}
