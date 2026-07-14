#include "vision_pipeline.h"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <stdexcept>
#include <algorithm>

VisionPipeline::VisionPipeline(const std::string& model_path,
                               const cv::Mat& K,
                               const cv::Mat& D,
                               const Eigen::Matrix3d& R_CB,
                               const Eigen::Vector3d& t_CB,
                               float conf_threshold,
                               float nms_threshold)
    : conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      K_(K.clone()),
      D_(D.clone()),
      R_CB_(R_CB),
      t_CB_(t_CB),
      env_(ORT_LOGGING_LEVEL_WARNING, "VisionPipeline"),
      session_options_() {
    
    // Set 2 threads for CPU inference to prevent CPU starvation on companion computer
    session_options_.SetIntraOpNumThreads(2);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
    } catch (const std::exception& e) {
        std::cerr << "[VisionPipeline] Failed to load ONNX model: " << e.what() << std::endl;
        throw;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Retrieve input node details
    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name_ptr = session_->GetInputNameAllocated(i, allocator);
        input_node_names_.push_back(std::string(input_name_ptr.get()));
        
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_ = tensor_info.GetShape();
    }

    // Retrieve output node details
    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
        output_node_names_.push_back(std::string(output_name_ptr.get()));
    }

    // Prepare const char* arrays for session->Run interface
    for (const auto& name : input_node_names_) {
        input_names_char_.push_back(name.c_str());
    }
    for (const auto& name : output_node_names_) {
        output_names_char_.push_back(name.c_str());
    }

    // Verify input dimensions
    if (input_node_dims_.size() < 4) {
        throw std::runtime_error("ONNX model input shape must be 4D (batch, channels, height, width)");
    }

    // Define 3D gate corners in local Gate frame (origin at center)
    // We model a square gate of size 1.5m x 1.5m.
    // Corners order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    float w_in = 0.75f;
    float h_in = 0.75f;
    object_points_ = {
        cv::Point3f(-w_in, -h_in, 0.0f), // Top-Left
        cv::Point3f( w_in, -h_in, 0.0f), // Top-Right
        cv::Point3f( w_in,  h_in, 0.0f), // Bottom-Right
        cv::Point3f(-w_in,  h_in, 0.0f)  // Bottom-Left
    };
}

bool VisionPipeline::process_frame(const cv::Mat& crop_img,
                                   float crop_x_offset,
                                   float crop_y_offset,
                                   float original_width,
                                   float original_height,
                                   const Eigen::Vector3d& gate_world_pos,
                                   const Eigen::Quaterniond& gate_world_rot,
                                   double timestamp,
                                   PoseMeasurement& measurement) {
    if (crop_img.empty()) {
        return false;
    }

    // ==========================================
    // Step 1: Preprocess ROI Crop (BGR -> Float RGB CHW)
    // ==========================================
    int target_w = 320;
    int target_h = 320;
    
    cv::Mat resized;
    cv::resize(crop_img, resized, cv::Size(target_w, target_h));
    
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    cv::Mat rgb_img;
    cv::cvtColor(float_img, rgb_img, cv::COLOR_BGR2RGB);

    // Flatten to planar CHW layout
    std::vector<float> input_tensor_values(1 * 3 * target_h * target_w);
    float* input_data = input_tensor_values.data();
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                input_data[c * target_h * target_w + h * target_w + w] = rgb_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    // ==========================================
    // Step 2: Run ONNX Runtime YOLO-Pose Inference
    // ==========================================
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> input_shape = {1, 3, target_h, target_w};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_char_.data(), &input_tensor, 1,
            output_names_char_.data(), output_names_char_.size());
    } catch (const std::exception& e) {
        std::cerr << "[VisionPipeline] Inference failed: " << e.what() << std::endl;
        return false;
    }

    if (output_tensors.empty()) return false;

    // ==========================================
    // Step 3: Postprocess output tensor & NMS
    // ==========================================
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    if (output_shape.size() < 3) return false;

    int num_channels = output_shape[1]; // 17
    int num_anchors = output_shape[2];  // 2100

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point2f>> all_keypoints;

    for (int j = 0; j < num_anchors; ++j) {
        float score = output_data[4 * num_anchors + j]; // Class score
        if (score >= conf_threshold_) {
            float cx = output_data[0 * num_anchors + j];
            float cy = output_data[1 * num_anchors + j];
            float w  = output_data[2 * num_anchors + j];
            float h  = output_data[3 * num_anchors + j];

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;

            // Clamping box to model bounds
            x1 = std::max(0.0f, std::min(x1, (float)target_w));
            y1 = std::max(0.0f, std::min(y1, (float)target_h));
            w  = std::max(0.0f, std::min(w, (float)target_w - x1));
            h  = std::max(0.0f, std::min(h, (float)target_h - y1));

            bboxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(w), static_cast<int>(h)));
            scores.push_back(score);

            // Extract the 4 corner keypoints
            std::vector<cv::Point2f> kpts;
            for (int k = 0; k < 4; ++k) {
                float kp_x = output_data[(5 + 3 * k) * num_anchors + j];
                float kp_y = output_data[(6 + 3 * k) * num_anchors + j];
                kpts.push_back(cv::Point2f(kp_x, kp_y));
            }
            all_keypoints.push_back(kpts);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, scores, conf_threshold_, nms_threshold_, nms_indices);

    if (nms_indices.empty()) return false;

    // Use the detection with the highest confidence
    int best_idx = nms_indices[0];
    float best_conf = scores[best_idx];

    // Scale and translate the 4 keypoints back to full-frame coordinates
    float scale_x = original_width / static_cast<float>(target_w);
    float scale_y = original_height / static_cast<float>(target_h);
    std::vector<cv::Point2f> scaled_keypoints(4);
    for (int k = 0; k < 4; ++k) {
        float sx = all_keypoints[best_idx][k].x * scale_x;
        float sy = all_keypoints[best_idx][k].y * scale_y;
        scaled_keypoints[k] = cv::Point2f(sx + crop_x_offset, sy + crop_y_offset);
    }

    // ==========================================
    // Step 4: Undistort corners & solve PnP
    // ==========================================
    std::vector<cv::Point2f> undistorted_keypoints;
    try {
        cv::fisheye::undistortPoints(scaled_keypoints, undistorted_keypoints, K_, D_, cv::Mat(), K_);
    } catch (const cv::Exception& e) {
        std::cerr << "[VisionPipeline] Equidistant undistort failed: " << e.what() << std::endl;
        return false;
    }

    cv::Mat rvec, tvec;
    bool pnp_success = false;
    try {
        pnp_success = cv::solvePnP(object_points_, undistorted_keypoints, K_, cv::Mat::zeros(4, 1, CV_64F),
                                   rvec, tvec, false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception& e) {
        std::cerr << "[VisionPipeline] solvePnP failed: " << e.what() << std::endl;
        return false;
    }

    if (!pnp_success) return false;

    // ==========================================
    // Step 5: Absolute Pose Reconstruction (Fusing Extrinsics)
    // ==========================================
    cv::Mat R_rel_mat;
    cv::Rodrigues(rvec, R_rel_mat);

    Eigen::Matrix3d R_rel;
    R_rel << R_rel_mat.at<double>(0, 0), R_rel_mat.at<double>(0, 1), R_rel_mat.at<double>(0, 2),
             R_rel_mat.at<double>(1, 0), R_rel_mat.at<double>(1, 1), R_rel_mat.at<double>(1, 2),
             R_rel_mat.at<double>(2, 0), R_rel_mat.at<double>(2, 1), R_rel_mat.at<double>(2, 2);

    Eigen::Vector3d t_rel(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // Drone rotation R_WB = R_world_gate * R_rel^T * R_CB
    Eigen::Matrix3d R_world_gate = gate_world_rot.toRotationMatrix();
    Eigen::Matrix3d R_meas = R_world_gate * R_rel.transpose() * R_CB_;
    
    // Drone position p_WB = gate_world_pos - R_WB * R_CB^T * (t_rel - t_CB)
    Eigen::Vector3d p_B_gate = R_CB_.transpose() * (t_rel - t_CB_);
    Eigen::Vector3d p_meas = gate_world_pos - R_meas * p_B_gate;

    // Output measurements
    measurement.timestamp = timestamp;
    measurement.position = p_meas;
    measurement.quaternion = Eigen::Quaterniond(R_meas).normalized();
    measurement.confidence = best_conf;

    return true;
}
