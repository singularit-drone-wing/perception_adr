#pragma once

#include "pipeline_utils.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>

class EKF {
public:
    // Constructor. Initializes EKF state and error covariance.
    EKF();
    ~EKF() = default;

    // Disallow copying for thread safety
    EKF(const EKF&) = delete;
    EKF& operator=(const EKF&) = delete;

    // Set/Get the current state
    void set_state(const KinematicState& state);
    KinematicState get_state(double timestamp) const;

    // Set/Get error covariance matrix
    void set_covariance(const Eigen::Matrix<double, 15, 15>& P);
    Eigen::Matrix<double, 15, 15> get_covariance() const;

    // Configure filter noise parameters
    void set_noise_parameters(double sigma_acc, double sigma_gyro,
                              double sigma_acc_bias, double sigma_gyro_bias,
                              double sigma_meas_pos, double sigma_meas_rot);

    // EKF Prediction Step (Thread 2):
    // Propagates the 10-DoF kinematic state forward in time using Runge-Kutta 4th Order (RK4).
    // Propagates the 15x15 error state covariance matrix P.
    // Inputs:
    //   - acc_raw: Raw 3D linear accelerometer reading in body frame.
    //   - gyro_raw: Raw 3D gyroscope reading in body frame.
    //   - dt: Timestep interval since last prediction.
    void predict(const Eigen::Vector3d& acc_raw, const Eigen::Vector3d& gyro_raw, double dt);

    // EKF Correction Step (Thread 3):
    // Corrects the accumulated state drift and sensor biases using visual global pose measurements.
    // Employs a delay-compensation scheme by utilizing the historical predicted state (from RingBuffer).
    // Inputs:
    //   - measurement: Timestamped global 6-DoF pose measurement (z_meas) from VisionPipeline.
    //   - historical_state: The historical kinematic state matching the camera capture timestamp (t_cam).
    // Returns: true if update succeeded, false otherwise.
    bool update(const PoseMeasurement& measurement, const KinematicState& historical_state);

private:
    mutable std::mutex mutex_;

    // Nominal State variables
    Eigen::Vector3d position_;       // p_WB (Position in World)
    Eigen::Vector3d velocity_;       // v_W (Velocity in World)
    Eigen::Quaterniond quaternion_; // q_WB (Attitude Rotating from Body to World)
    Eigen::Vector3d acc_bias_;       // b_a (Accelerometer Bias)
    Eigen::Vector3d gyro_bias_;      // b_g (Gyroscope Bias)

    // 15x15 Error State Covariance Matrix
    Eigen::Matrix<double, 15, 15> P_;

    // Continuous Noise parameter SDs
    double sigma_acc_;
    double sigma_gyro_;
    double sigma_acc_bias_;
    double sigma_gyro_bias_;

    // Measurement Noise SDs
    double sigma_meas_pos_;
    double sigma_meas_rot_;

    // Constant Gravity vector (defined as Z-down positive in NED World Frame)
    const Eigen::Vector3d gravity_;
};
