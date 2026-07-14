#include "ekf.h"
#include <iostream>

// Skew-symmetric matrix helper
static Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<   0.0,  -v.z(),  v.y(),
          v.z(),   0.0,  -v.x(),
         -v.y(),  v.x(),   0.0;
    return m;
}

EKF::EKF()
    : position_(Eigen::Vector3d::Zero()),
      velocity_(Eigen::Vector3d::Zero()),
      quaternion_(Eigen::Quaterniond::Identity()),
      acc_bias_(Eigen::Vector3d::Zero()),
      gyro_bias_(Eigen::Vector3d::Zero()),
      gravity_(0.0, 0.0, 9.81) { // Positive 9.81 m/s^2 along Z-down in NED World frame
    
    // Initialize error covariance P to small values
    P_ = Eigen::Matrix<double, 15, 15>::Identity() * 0.01;
    // Set larger uncertainty for velocities and biases initially
    P_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.1; // Velocity covariance
    P_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 0.05; // Acc bias covariance
    P_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 0.01; // Gyro bias covariance

    // Set default noise parameters
    sigma_acc_ = 0.1;        // m/s^2
    sigma_gyro_ = 0.01;      // rad/s
    sigma_acc_bias_ = 0.001;  // m/s^3
    sigma_gyro_bias_ = 0.0001; // rad/s^2
    
    sigma_meas_pos_ = 0.05;  // m
    sigma_meas_rot_ = 0.02;  // rad
}

void EKF::set_state(const KinematicState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_ = state.position;
    velocity_ = state.velocity;
    quaternion_ = state.quaternion.normalized();
    acc_bias_ = state.acc_bias;
    gyro_bias_ = state.gyro_bias;
}

KinematicState EKF::get_state(double timestamp) const {
    std::lock_guard<std::mutex> lock(mutex_);
    KinematicState state;
    state.timestamp = timestamp;
    state.position = position_;
    state.velocity = velocity_;
    state.quaternion = quaternion_;
    state.acc_bias = acc_bias_;
    state.gyro_bias = gyro_bias_;
    return state;
}

void EKF::set_covariance(const Eigen::Matrix<double, 15, 15>& P) {
    std::lock_guard<std::mutex> lock(mutex_);
    P_ = P;
}

Eigen::Matrix<double, 15, 15> EKF::get_covariance() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return P_;
}

void EKF::set_noise_parameters(double sigma_acc, double sigma_gyro,
                              double sigma_acc_bias, double sigma_gyro_bias,
                              double sigma_meas_pos, double sigma_meas_rot) {
    std::lock_guard<std::mutex> lock(mutex_);
    sigma_acc_ = sigma_acc;
    sigma_gyro_ = sigma_gyro;
    sigma_acc_bias_ = sigma_acc_bias;
    sigma_gyro_bias_ = sigma_gyro_bias;
    sigma_meas_pos_ = sigma_meas_pos;
    sigma_meas_rot_ = sigma_meas_rot;
}

// -----------------------------------------------------------------
// EKF Prediction Step (Thread 2)
// -----------------------------------------------------------------
void EKF::predict(const Eigen::Vector3d& acc_raw, const Eigen::Vector3d& gyro_raw, double dt) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Compensate sensor measurements for estimated biases
    Eigen::Vector3d a_corrected = acc_raw - acc_bias_;
    Eigen::Vector3d w_corrected = gyro_raw - gyro_bias_;

    // 2. Nominal State Propagation using Runge-Kutta 4th Order (RK4)
    // Derivative definition: 
    // dp/dt = v
    // dv/dt = R(q) * a_corrected + gravity
    // dq/dt = 0.5 * q * w_corrected (pure quaternion multiplication)
    auto compute_deriv = [this](const Eigen::Vector3d& p, const Eigen::Vector3d& v, 
                                 const Eigen::Quaterniond& q, const Eigen::Vector3d& a, 
                                 const Eigen::Vector3d& w, Eigen::Vector3d& dp, 
                                 Eigen::Vector3d& dv, Eigen::Quaterniond& dq) {
        dp = v;
        dv = q * a + gravity_;
        
        // Pure quaternion representation of angular rates
        Eigen::Quaterniond w_pure(0.0, w.x(), w.y(), w.z());
        dq = q * w_pure;
        dq.coeffs() *= 0.5;
    };

    Eigen::Vector3d dp1, dv1, dp2, dv2, dp3, dv3, dp4, dv4;
    Eigen::Quaterniond dq1, dq2, dq3, dq4;

    // k1
    compute_deriv(position_, velocity_, quaternion_, a_corrected, w_corrected, dp1, dv1, dq1);

    // k2 (evaluated at t + dt/2)
    Eigen::Vector3d p2 = position_ + dp1 * (dt / 2.0);
    Eigen::Vector3d v2 = velocity_ + dv1 * (dt / 2.0);
    Eigen::Quaterniond q2;
    q2.coeffs() = quaternion_.coeffs() + dq1.coeffs() * (dt / 2.0);
    q2.normalize();
    compute_deriv(p2, v2, q2, a_corrected, w_corrected, dp2, dv2, dq2);

    // k3 (evaluated at t + dt/2)
    Eigen::Vector3d p3 = position_ + dp2 * (dt / 2.0);
    Eigen::Vector3d v3 = velocity_ + dv2 * (dt / 2.0);
    Eigen::Quaterniond q3;
    q3.coeffs() = quaternion_.coeffs() + dq2.coeffs() * (dt / 2.0);
    q3.normalize();
    compute_deriv(p3, v3, q3, a_corrected, w_corrected, dp3, dv3, dq3);

    // k4 (evaluated at t + dt)
    Eigen::Vector3d p4 = position_ + dp3 * dt;
    Eigen::Vector3d v4 = velocity_ + dv3 * dt;
    Eigen::Quaterniond q4;
    q4.coeffs() = quaternion_.coeffs() + dq3.coeffs() * dt;
    q4.normalize();
    compute_deriv(p4, v4, q4, a_corrected, w_corrected, dp4, dv4, dq4);

    // Weighted RK4 integration summation
    position_ += (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
    velocity_ += (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
    
    Eigen::Quaterniond q_next_raw;
    q_next_raw.coeffs() = quaternion_.coeffs() + (dt / 6.0) * (dq1.coeffs() + 2.0 * dq2.coeffs() + 2.0 * dq3.coeffs() + dq4.coeffs());
    quaternion_ = q_next_raw.normalized();

    // 3. Covariance Propagation
    // Error State Transition Matrix F (15x15)
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
    
    // Position error derivative: d(delta_p)/dt = delta_v
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    
    // Velocity error derivative: d(delta_v)/dt = -R_WB * [a_corrected]x * delta_theta - R_WB * delta_ba
    Eigen::Matrix3d R_WB = quaternion_.toRotationMatrix();
    F.block<3, 3>(3, 6) = -R_WB * skew_symmetric(a_corrected) * dt;
    F.block<3, 3>(3, 9) = -R_WB * dt;

    // Orientation error derivative: d(delta_theta)/dt = -[w_corrected]x * delta_theta - delta_bg
    F.block<3, 3>(6, 6) = (Eigen::Matrix3d::Identity() - skew_symmetric(w_corrected) * dt);
    F.block<3, 3>(6, 12) = -Eigen::Matrix3d::Identity() * dt;

    // Construct discrete-time process noise covariance Q_d
    Eigen::Matrix<double, 15, 15> Q = Eigen::Matrix<double, 15, 15>::Zero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_acc_ * sigma_acc_ * dt * dt); // Position random noise
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_acc_ * sigma_acc_ * dt);       // Velocity random noise
    Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (sigma_gyro_ * sigma_gyro_ * dt);     // Orientation random noise
    Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (sigma_acc_bias_ * sigma_acc_bias_ * dt); // Acc bias random walk
    Q.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * (sigma_gyro_bias_ * sigma_gyro_bias_ * dt); // Gyro bias random walk

    // Propagate covariance
    P_ = F * P_ * F.transpose() + Q;
}

// -----------------------------------------------------------------
// EKF Correction Step (Thread 3)
// -----------------------------------------------------------------
bool EKF::update(const PoseMeasurement& measurement, const KinematicState& historical_state) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Compute innovation residuals in the system error space
    // Position residual: y_p = p_meas - p_pred
    Eigen::Vector3d y_pos = measurement.position - historical_state.position;

    // Orientation residual: y_theta = 2.0 * vec(q_pred_inv * q_meas)
    Eigen::Quaterniond q_err = historical_state.quaternion.inverse() * measurement.quaternion;
    
    // Choose shorter rotation path on the unit sphere
    if (q_err.w() < 0.0) {
        q_err.coeffs() = -q_err.coeffs();
    }
    // Small angle approximation error vector
    Eigen::Vector3d y_rot = 2.0 * q_err.vec();

    Eigen::Matrix<double, 6, 1> y;
    y.segment<3>(0) = y_pos;
    y.segment<3>(3) = y_rot;

    // 2. Construct measurement Jacobian H (6x15)
    // Residual directly observes error state position (indices 0-2) and orientation error (indices 6-8)
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity(); // d(y_pos)/d(delta_p)
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity(); // d(y_rot)/d(delta_theta)

    // 3. Construct measurement noise covariance matrix R (6x6)
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
    R.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (sigma_meas_pos_ * sigma_meas_pos_);
    R.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (sigma_meas_rot_ * sigma_meas_rot_);

    // 4. Compute Kalman Gain K (15x6)
    Eigen::Matrix<double, 6, 6> S = H * P_ * H.transpose() + R;
    Eigen::Matrix<double, 15, 6> K = P_ * H.transpose() * S.inverse();

    // 5. Compute Error State correction: delta_x = K * y
    Eigen::Matrix<double, 15, 1> delta_x = K * y;

    // 6. Apply correction to Nominal State
    position_ += delta_x.segment<3>(0);
    velocity_ += delta_x.segment<3>(3);
    
    // Apply orientation small-angle rotation: q_next = q * exp(delta_theta)
    Eigen::Vector3d delta_theta = delta_x.segment<3>(6);
    Eigen::Vector3d half_theta = 0.5 * delta_theta;
    Eigen::Quaterniond q_exp(1.0, half_theta.x(), half_theta.y(), half_theta.z());
    quaternion_ = (quaternion_ * q_exp).normalized();

    // Apply biases updates
    acc_bias_ += delta_x.segment<3>(9);
    gyro_bias_ += delta_x.segment<3>(12);

    // 7. Update Error Covariance P using the numerically stable Joseph Form:
    // P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
    Eigen::Matrix<double, 15, 15> I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
    P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();

    // Enforce symmetry in covariance matrix to prevent numerical divergence
    P_ = 0.5 * (P_ + P_.transpose());

    return true;
}
