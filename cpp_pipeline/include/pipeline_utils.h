#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <optional>
#include <chrono>
#include <algorithm>

// ==========================================
// 1. Common Types & Structures
// ==========================================

struct DetectionResult {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> keypoints; // 4 corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
};

struct PoseMeasurement {
    double timestamp;
    Eigen::Vector3d position;       // Estimated global position of drone in world (p_meas)
    Eigen::Quaterniond quaternion; // Estimated global attitude of drone in world (q_meas)
    double confidence;             // Visual detection confidence score
};

struct KinematicState {
    double timestamp;
    Eigen::Vector3d position;       // p_WB (position in world)
    Eigen::Vector3d velocity;       // v_W (velocity in world)
    Eigen::Quaterniond quaternion; // q_WB (attitude in world)
    Eigen::Vector3d acc_bias;       // b_a (accelerometer bias)
    Eigen::Vector3d gyro_bias;      // b_g (gyroscope bias)

    Eigen::Matrix<double, 16, 1> to_vector() const {
        Eigen::Matrix<double, 16, 1> x;
        x.segment<3>(0) = position;
        x.segment<3>(3) = velocity;
        x(6) = quaternion.w(); x(7) = quaternion.x(); x(8) = quaternion.y(); x(9) = quaternion.z();
        x.segment<3>(10) = acc_bias;
        x.segment<3>(13) = gyro_bias;
        return x;
    }

    static KinematicState from_vector(double ts, const Eigen::Matrix<double, 16, 1>& x) {
        KinematicState s;
        s.timestamp = ts;
        s.position = x.segment<3>(0);
        s.velocity = x.segment<3>(3);
        s.quaternion = Eigen::Quaterniond(x(6), x(7), x(8), x(9)).normalized();
        s.acc_bias = x.segment<3>(10);
        s.gyro_bias = x.segment<3>(13);
        return s;
    }
};

// ==========================================
// 2. Thread-Safe Queue (Vision -> EKF)
// ==========================================

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        cond_var_.notify_one();
    }

    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    std::optional<T> pop_with_timeout(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cond_var_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_var_;
};

// ==========================================
// 3. EKF State Ring Buffer (Delay Compensation)
// ==========================================

class RingBuffer {
public:
    explicit RingBuffer(size_t max_size = 2000) : max_size_(max_size) {}
    ~RingBuffer() = default;

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    void insert(const KinematicState& state) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!buffer_.empty() && state.timestamp <= buffer_.back().timestamp) {
            return; // Ignore out-of-order predictions
        }
        buffer_.push_back(state);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }

    std::optional<KinematicState> get_state_at(double timestamp) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) return std::nullopt;

        if (timestamp <= buffer_.front().timestamp) return buffer_.front();
        if (timestamp >= buffer_.back().timestamp) return buffer_.back();

        auto it = std::lower_bound(buffer_.begin(), buffer_.end(), timestamp,
            [](const KinematicState& state, double ts) {
                return state.timestamp < ts;
            });

        if (it == buffer_.end()) return buffer_.back();
        if (it->timestamp == timestamp) return *it;

        const auto& s2 = *it;
        const auto& s1 = *(it - 1);

        double t1 = s1.timestamp;
        double t2 = s2.timestamp;
        double alpha = (timestamp - t1) / (t2 - t1);

        KinematicState interpolated;
        interpolated.timestamp = timestamp;
        interpolated.position = (1.0 - alpha) * s1.position + alpha * s2.position;
        interpolated.velocity = (1.0 - alpha) * s1.velocity + alpha * s2.velocity;
        interpolated.quaternion = s1.quaternion.slerp(alpha, s2.quaternion).normalized();
        interpolated.acc_bias = (1.0 - alpha) * s1.acc_bias + alpha * s2.acc_bias;
        interpolated.gyro_bias = (1.0 - alpha) * s1.gyro_bias + alpha * s2.gyro_bias;

        return interpolated;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

private:
    size_t max_size_;
    mutable std::mutex mutex_;
    std::deque<KinematicState> buffer_;
};
