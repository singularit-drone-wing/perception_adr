#pragma once

#include "pipeline_utils.h"
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>

class VisionPipeline {
public:
    // Constructor initializes both ONNX Runtime for YOLO-Pose and PnP calibration matrices
    VisionPipeline(const std::string& model_path,
                   const cv::Mat& K,
                   const cv::Mat& D,
                   const Eigen::Matrix3d& R_CB,
                   const Eigen::Vector3d& t_CB,
                   float conf_threshold = 0.5f,
                   float nms_threshold = 0.45f);
    
    ~VisionPipeline() = default;

    // Disable copy/move constructors
    VisionPipeline(const VisionPipeline&) = delete;
    VisionPipeline& operator=(const VisionPipeline&) = delete;

    // Main vision pipeline entry point:
    // 1. Preprocesses ROI crop
    // 2. Runs YOLO-Pose ONNX Runtime inference
    // 3. Re-scales keypoints to full-frame space
    // 4. Undistorts corners and solves PnP using IPPE solver
    // 5. Transforms camera-relative pose to absolute global drone pose
    // Returns: true if a gate was successfully detected and pose resolved.
    bool process_frame(const cv::Mat& crop_img,
                       float crop_x_offset,
                       float crop_y_offset,
                       float original_width,
                       float original_height,
                       const Eigen::Vector3d& gate_world_pos,
                       const Eigen::Quaterniond& gate_world_rot,
                       double timestamp,
                       PoseMeasurement& measurement);

private:
    // Vision & Solver settings
    float conf_threshold_;
    float nms_threshold_;

    // Camera calibration
    cv::Mat K_; // Camera Matrix (3x3)
    cv::Mat D_; // Distortion Coefficients (4x1)
    
    // Extrinsics transforming from Body (IMU) to Camera frame
    Eigen::Matrix3d R_CB_;
    Eigen::Vector3d t_CB_;

    // Local 3D gate corners in local Gate frame (origin at center)
    std::vector<cv::Point3f> object_points_;

    // ONNX Runtime Session members
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<const char*> input_names_char_;
    std::vector<const char*> output_names_char_;
    std::vector<int64_t> input_node_dims_;
};
