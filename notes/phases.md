# Development Phases & Roadmap

This document outlines the step-by-step development phases, milestone checklists, and technical execution details of the Autonomous Drone Racing Perception & State Estimation (VIO) pipeline.

---

## Milestone Summary

- [x] **Phase 1: Synthetic Dataset & YOLO-Pose Training**
  - [x] Write Python synthetic data generator using raw UZH background frames.
  - [x] Auto-project 3D gate corners using Kannala-Brandt fisheye camera distortion.
  - [x] Write training script (`simulation/train_yolo.py`) to train the model and export to ONNX.

- [x] **Phase 2: C++ Pipeline Core**
  - [x] Implement thread-safe queues and ring buffers.
  - [x] Integrate ONNX Runtime for YOLO-Pose inference.
  - [x] Implement OpenCV PnP localization solver.

- [x] **Phase 3: EKF State Fusion**
  - [x] Implement RK4 IMU kinematics integrator.
  - [x] Implement EKF update step with delay compensation.

- [x] **Phase 4: Closed-Loop Simulation Testing**
  - [x] Set up UDP socket communications.
  - [x] Run test flights in Python simulator using C++ estimates.

---

## Phase 1: Synthetic Dataset & YOLO-Pose Training

### Technical Details

#### 1. Synthetic Dataset Generation (`simulation/generate_synthetic_gates.py`)
To train YOLO-Pose without manual labeling, a training dataset is synthesized directly from the UZH-FPV background images:
* **Backgrounds**: Raw frames from the UZH dataset where the gate is not visible or arbitrary scenes.
* **3D Gate Projection**: A virtual square gate ($1.5\text{m} \times 1.5\text{m}$ with $0.08\text{m}$ border thickness) is placed in 3D relative to the camera frame ($Z$-forward, $X$-right, $Y$-down).
* **Fisheye Lens Distortion**: To match the real Davis240C camera lens, 3D corners are projected onto the 2D plane using the **Kannala-Brandt (equidistant) distortion model**:
  $$\theta = \arctan(r)$$
  $$\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)$$
  where $k_1, k_2, k_3, k_4$ are calibrated distortion coefficients.
* **Augmentations**: Randomizes rotation, translation, gate color intensity, and applies Gaussian/motion blur to simulate high-speed flight.
* **Output**: Writes 1,200 training and 150 validation images in YOLO-Pose format (`class_idx x_center y_center w h kp1_x kp1_y kp1_v ...`) to `datasets/yolo_gate/`.

#### 2. YOLO-Pose Training (`simulation/train_yolo.py`)
* Loads lightweight `yolo11n-pose.pt` base model.
* Trains on synthetic dataset for 30 epochs (automatically selecting CUDA if available).
* Exports final model to **ONNX format** with dynamic axes, ready for loading by the C++ pipeline.

##### Training Metrics (30 Epochs)

| Epoch | Train Box Loss | Val Box Loss | Train Keypoint Loss | Val Keypoint Loss | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.1984 | 0.7089 | 3.6466 | 2.2203 | 0.9572 | 0.8129 | 0.9572 | 0.5018 |
| 5 | 0.6624 | 0.5227 | 0.4967 | 0.2752 | 0.9944 | 0.8804 | 0.9944 | 0.9711 |
| 10 | 0.5578 | 0.4340 | 0.3349 | 0.1506 | 0.9949 | 0.9102 | 0.9949 | 0.9914 |
| 15 | 0.5011 | 0.3658 | 0.2624 | 0.1113 | 0.9950 | 0.9302 | 0.9950 | 0.9926 |
| 20 | 0.4392 | 0.3146 | 0.2068 | 0.0816 | 0.9950 | 0.9595 | 0.9950 | 0.9929 |
| 25 | 0.2805 | 0.2742 | 0.0677 | 0.0631 | 0.9950 | 0.9683 | 0.9950 | 0.9944 |
| 30 | 0.2343 | 0.2322 | 0.0540 | 0.0465 | 0.9950 | 0.9781 | 0.9950 | 0.9950 |

---

## Phase 2: C++ Pipeline Core

### Technical Details

The core C++ components are implemented in `cpp_pipeline/`.

#### 1. File-Wise Architecture
* **`pipeline_utils.h` (Shared Types & Concurrency Buffers)**:
  * `DetectionResult`: `cv::Rect bbox`, `confidence`, and 4 corner `keypoints`.
  * `PoseMeasurement`: 3D global position (`Eigen::Vector3d`) and attitude quaternion (`Eigen::Quaterniond`).
  * `KinematicState`: 16D EKF nominal state vectors.
  * `ThreadSafeQueue<T>`: Mutex-locked queue with `std::condition_variable` notification for thread-safe pushing/popping.
  * `RingBuffer`: Monotonic timestamp-indexed ring buffer with binary search lookup (`std::lower_bound`) and SLERP attitude interpolation for latency compensation.
* **`vision_pipeline.cpp` (Unified Vision Engine)**:
  * Manages ONNX Runtime session, image preprocessing (320x320 BGR-to-RGB, normalization), NMS postprocessing, Kannala-Brandt keypoint undistortion, and IPPE PnP relative pose calculation ($R_{rel}, t_{rel}$).
  * `process_frame_with_gate_matching(...)`: Matches candidates against track map poses using closest-state nearest-gate lookup with distance outlier rejection ($> 15\text{m}$).
* **`verify_pipeline.cpp` (Verification Runner)**:
  * Test script loading sample frame `image_0_322.png`, verifying PnP solving, gate lookup, and RingBuffer SLERP interpolation without network execution.

---

## Phase 3: EKF State Fusion

### Technical Details

Implemented in `cpp_pipeline/include/ekf.h` and `cpp_pipeline/src/ekf.cpp`.

#### 1. State Representation & Covariance
* **Nominal State (16D)**: $x = [p_{WB}, v_W, q_{WB}, b_a, b_g]^T$
* **Error State (15D)**: $\delta x = [\delta p, \delta v, \delta \theta, \delta b_a, \delta b_g]^T$
* **Covariance Matrix**: Maintains 15x15 error covariance matrix $P$.

#### 2. Runge-Kutta 4th Order Kinematic Integration (`predict()`)
Propagates state forward over timestep $dt$:
* Zero-mean bias-corrected inputs: $a_{corr} = a_{raw} - b_a$, $\omega_{corr} = \omega_{raw} - b_g$.
* State derivatives: $\dot{p} = v$, $\dot{v} = R(q_{WB}) a_{corr} + g_W$, $\dot{q} = \frac{1}{2} q_{WB} \otimes \omega_{corr}$.
* Covariance propagation: $P_k = F P_{k-1} F^T + Q$ with error transition matrix $F$ and discrete process noise $Q$.

#### 3. Joseph-Form Measurement Update (`update()`)
* **Innovation Residual (6D)**: $y_p = p_{meas} - p_{hist}$, $y_\theta = 2 \cdot \text{vec}(q_{hist}^{-1} \otimes q_{meas})$.
* **Kalman Gain**: $K = P H^T (H P H^T + R)^{-1}$.
* **Joseph-Form Covariance Update**:
  $$P \leftarrow (I - KH) P (I - KH)^T + K R K^T$$
  $$P \leftarrow \frac{1}{2} (P + P^T)$$

---

## Phase 4: Closed-Loop Simulation Testing

### Technical Details

Established real-time closed-loop IPC using production node `cpp_pipeline/src/main.cpp` and Python flight harness `simulation/simulate_drone.py`.

#### 1. Concurrent Threading Architecture
* **Thread 2 (IMU Receiver & Predictor / 500-1000 Hz)**: Listens on UDP port `12345`. Runs `ekf.predict`, stores predictions in `RingBuffer`, and dispatches camera frames to `camera_queue`.
* **Thread 1 (Vision Pipeline / 30-60 Hz)**: Pops frames, runs YOLO ONNX inference, undistorts keypoints, solves IPPE PnP, performs nearest-gate lookup, and pushes `PoseMeasurement` to EKF correction queue.
* **Thread 3 (EKF Update & UDP Sender / ~10 Hz)**: Pops measurements, fetches historical latency-compensated states from `RingBuffer`, executes `ekf.update`, and transmits state datagrams back to Python on UDP port `12346`.

#### 2. UDP Packet Specifications
* **IMU Datagram (`'I'`)**: 65 bytes (`<c d d d d d d d`).
* **Camera Datagram (`'C'`)**: 13 bytes header (`<c d I`) + binary JPEG payload.
* **State Datagram (`'S'`)**: 137 bytes (`<c d d d d d d d d d d d d d d d d d`).
