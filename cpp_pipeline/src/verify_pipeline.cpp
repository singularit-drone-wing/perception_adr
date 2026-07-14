#include "vision_pipeline.h"
#include "pipeline_utils.h"
#include "ekf.h"
#include <iostream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv) {
    std::cout << "=========================================" << std::endl;
    std::cout << "Autonomous Drone Racing C++ Pipeline Verification" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Define file paths (relative to cpp_pipeline/build/ where the binary is run)
    std::string model_path = "../../weights/best.onnx";
    std::string image_path = "../../datasets/uzh-fpv-indoor-forward-davis3/img/image_0_322.png";

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        image_path = argv[2];
    }

    // 1. Initialize PnP Solver calibration parameters
    // Camera Intrinsic Matrix K (3x3)
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        172.98992850734132,   0.0,                163.33639726024606,
          0.0,                172.98303181090185, 134.99537889030861,
          0.0,                  0.0,                1.0);

    // Equidistant (Kannala-Brandt) Distortion Coefficients D (4x1)
    cv::Mat D = (cv::Mat_<double>(4, 1) << 
        -0.027576733308582076,
        -0.006593578674675004,
         0.0008566938165177085,
        -0.00030899587045247486);

    // Extrinsics transforming from Body (IMU) to Camera frame
    Eigen::Matrix3d R_CB;
    R_CB <<  0.9999711474430529,     0.0013817010649267755, -0.007469617365767657,
            -0.0014085305353606873,  0.9999925720306121,    -0.00358774655345255,
             0.007464604688444933,   0.0035981642219379494,  0.9999656658561218;

    Eigen::Vector3d t_CB(0.00018050225881571712, -0.004316353415695194, -0.027547385763471585);

    // 2. Initialize consolidated VisionPipeline
    std::cout << "[Init] Loading VisionPipeline (YOLO-Pose & PnP) with model: " << model_path << std::endl;
    std::unique_ptr<VisionPipeline> pipeline;
    try {
        pipeline = std::make_unique<VisionPipeline>(model_path, K, D, R_CB, t_CB, 0.5f, 0.45f);
        std::cout << "[Init] VisionPipeline initialized successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to initialize VisionPipeline: " << e.what() << std::endl;
        return -1;
    }

    // 3. Load input validation frame
    std::cout << "[IO] Reading test image from: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[Error] Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "[IO] Image loaded successfully. Dimensions: " << img.cols << "x" << img.rows << std::endl;

    // 4. Run the VisionPipeline
    // For verification, we assume the crop is the full frame (no adaptive cropping offsets)
    float crop_x_offset = 0.0f;
    float crop_y_offset = 0.0f;
    float original_width = static_cast<float>(img.cols);
    float original_height = static_cast<float>(img.rows);
    double timestamp = 12345.6789; // Mock frame timestamp

    // Define target gate global pose (dummy coordinates for absolute reference frame testing)
    Eigen::Vector3d gate_world_pos(0.0, 0.0, 3.0); // 3.0m along world Z axis
    Eigen::Quaterniond gate_world_rot = Eigen::Quaterniond::Identity(); // Aligned with world axes

    std::cout << "[Pipeline] Processing frame..." << std::endl;
    PoseMeasurement measurement;
    bool success = pipeline->process_frame(
        img, crop_x_offset, crop_y_offset, original_width, original_height,
        gate_world_pos, gate_world_rot, timestamp, measurement);

    if (success) {
        std::cout << "[Pipeline] Success! Gate detected and pose resolved." << std::endl;
        std::cout << "  - Visual Confidence: " << std::fixed << std::setprecision(2) << measurement.confidence * 100.0 << "%" << std::endl;
        std::cout << "  - Estimated Drone Position (p_meas): [" 
                  << std::setprecision(4) << measurement.position.x() << ", " 
                  << measurement.position.y() << ", " 
                  << measurement.position.z() << "] meters" << std::endl;
        std::cout << "  - Estimated Drone Quaternion (q_meas) [w, x, y, z]: [" 
                  << measurement.quaternion.w() << ", " 
                  << measurement.quaternion.x() << ", " 
                  << measurement.quaternion.y() << ", " 
                  << measurement.quaternion.z() << "]" << std::endl;
    } else {
        std::cerr << "[Pipeline] Error: Vision pipeline failed to resolve pose. No gate detected above threshold." << std::endl;
    }

    // 4b. Run the VisionPipeline with Gate Matching
    std::cout << "[Pipeline] Running Gate Matching with pre-loaded map..." << std::endl;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> gate_map = {
        { Eigen::Vector3d(10.0, 10.0, 10.0), Eigen::Quaterniond::Identity() }, // Gate 0 (far away)
        { Eigen::Vector3d(0.0, 0.0, 3.0), Eigen::Quaterniond::Identity() },    // Gate 1 (correct gate)
        { Eigen::Vector3d(-5.0, 0.0, 5.0), Eigen::Quaterniond::Identity() }    // Gate 2 (wrong gate)
    };
    Eigen::Vector3d predicted_drone_pos(0.0, 0.0, 0.0);
    double max_match_distance = 15.0; // 15 meters threshold

    PoseMeasurement matched_measurement;
    bool match_success = pipeline->process_frame_with_gate_matching(
        img, crop_x_offset, crop_y_offset, original_width, original_height,
        gate_map, predicted_drone_pos, timestamp, max_match_distance, matched_measurement);

    if (match_success) {
        std::cout << "[Pipeline] Gate Matching Success!" << std::endl;
        std::cout << "  - Visual Confidence: " << std::fixed << std::setprecision(2) << matched_measurement.confidence * 100.0 << "%" << std::endl;
        std::cout << "  - Matched Drone Position (p_meas): [" 
                  << std::setprecision(4) << matched_measurement.position.x() << ", " 
                  << matched_measurement.position.y() << ", " 
                  << matched_measurement.position.z() << "] meters" << std::endl;
    } else {
        std::cerr << "[Pipeline] Error: Gate matching failed." << std::endl;
    }

    // 5. Verify concurrent structures (ThreadSafeQueue and RingBuffer)
    std::cout << "[Test] Verifying RingBuffer functionality..." << std::endl;
    RingBuffer ring_buffer(10);
    
    // Insert some dummy historical states
    for (int i = 0; i < 5; ++i) {
        KinematicState state;
        state.timestamp = 100.0 + i * 0.1; // 10Hz sampling
        state.position = Eigen::Vector3d(i * 0.5, 0.0, 0.0);
        state.velocity = Eigen::Vector3d(5.0, 0.0, 0.0);
        state.quaternion = Eigen::Quaterniond::Identity();
        state.acc_bias = Eigen::Vector3d::Zero();
        state.gyro_bias = Eigen::Vector3d::Zero();
        ring_buffer.insert(state);
    }
    
    // Query at an intermediate timestamp (100.25) to test interpolation
    double query_ts = 100.25;
    auto interp_state = ring_buffer.get_state_at(query_ts);
    if (interp_state) {
        std::cout << "  - Interpolation at t=" << query_ts << " succeeded." << std::endl;
        std::cout << "  - Interpolated Position (expected: [1.25, 0, 0]): [" 
                  << interp_state->position.x() << ", " 
                  << interp_state->position.y() << ", " 
                  << interp_state->position.z() << "]" << std::endl;
    } else {
        std::cerr << "  - Interpolation failed!" << std::endl;
    }

    // 6. Verify EKF functionality (propagation and correction)
    std::cout << "[Test] Verifying EKF propagation and update..." << std::endl;
    EKF ekf;

    // Set initial kinematic state
    KinematicState init_state;
    init_state.timestamp = timestamp;
    init_state.position = Eigen::Vector3d(0.0, 0.0, 0.0);
    init_state.velocity = Eigen::Vector3d(0.1, 0.0, 0.0);
    init_state.quaternion = Eigen::Quaterniond::Identity();
    init_state.acc_bias = Eigen::Vector3d(0.01, -0.01, 0.02);
    init_state.gyro_bias = Eigen::Vector3d(0.001, -0.001, 0.002);
    ekf.set_state(init_state);

    // Predict forward by 10ms with mock IMU readings (accel includes gravity, bias, and positive vertical acceleration)
    Eigen::Vector3d mock_acc(0.1, 0.0, 9.81 + 0.2 + 0.01); 
    Eigen::Vector3d mock_gyro(0.01, -0.01, 0.02);
    double predict_dt = 0.01;
    ekf.predict(mock_acc, mock_gyro, predict_dt);

    auto pred_state = ekf.get_state(timestamp + predict_dt);
    std::cout << "  - EKF Predict Position: [" 
              << pred_state.position.x() << ", " 
              << pred_state.position.y() << ", " 
              << pred_state.position.z() << "]" << std::endl;

    // Apply EKF correction update using the vision measurement
    if (success) {
        bool update_success = ekf.update(measurement, pred_state);
        if (update_success) {
            auto corrected_state = ekf.get_state(timestamp + predict_dt);
            std::cout << "  - EKF Update Success!" << std::endl;
            std::cout << "  - Corrected Position: [" 
                      << corrected_state.position.x() << ", " 
                      << corrected_state.position.y() << ", " 
                      << corrected_state.position.z() << "]" << std::endl;
            std::cout << "  - Corrected Acc Bias: [" 
                      << corrected_state.acc_bias.x() << ", " 
                      << corrected_state.acc_bias.y() << ", " 
                      << corrected_state.acc_bias.z() << "]" << std::endl;
        } else {
            std::cerr << "  - EKF Update failed!" << std::endl;
        }
    }

    std::cout << "=========================================" << std::endl;
    std::cout << "Verification Complete!" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
