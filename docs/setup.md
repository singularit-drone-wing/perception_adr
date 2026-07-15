# Setup and Replication Guide

This guide details the steps to replicate, build, and test the perception and state estimation pipeline on your local system.

---

## 1. System Prerequisites

The C++ perception engine uses POSIX networking headers (`<sys/socket.h>`, `<netinet/in.h>`, etc.) for UDP communications. 

* **Linux (Ubuntu/Debian)**: Fully supported natively.
* **Windows**: Because of POSIX dependencies, **WSL2 (Windows Subsystem for Linux)** is the recommended environment. Running natively on Windows MSVC/MinGW requires porting the socket code to Winsock2 (`<winsock2.h>`).

### Install System Libraries (Linux / WSL2)
Install CMake, OpenCV, ONNX Runtime, and Eigen3 headers:
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libonnxruntime-dev libeigen3-dev
```

---

## 2. Dataset Download & Configuration

1. Visit the [UZH-FPV Quadcopter Dataset Repository](https://fpv.ifi.uzh.ch/datasets/).
2. Download the **Indoor Forward-Facing (DAVIS-3 APS) Baseline** sequence dataset.
3. Extract the downloaded sequence files into the `datasets/` directory at the root of this project, ensuring the following directory layout:

```text
datasets/
└── uzh-fpv-indoor-forward-davis3/
    ├── calib/           # Camera & IMU calibration parameters (YAML)
    ├── img/             # Folder containing raw DAVIS240C frame PNGs
    ├── imu.txt          # High frequency 500Hz Accelerometer & Gyro logs
    ├── groundtruth.txt  # Leica-sync 6DoF Ground Truth poses
    ├── events.txt       # Neuromorphic event stream logs (optional)
    ├── images.txt       # Frame timestamp-to-file index mapping
    └── leica.txt        # Raw Leica total station records
```

---

## 3. Python Environment & Dependencies

Configure a virtual environment and install the required modules listed in `requirements.txt` (which includes OpenCV, ONNX, PyTorch CPU, and Ultralytics):

### Linux / WSL2
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Alternative)
*If you are running the Python client directly on Windows, activate the environment using:*
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. YOLO-Pose Model Weights & Optional Training

To allow immediate testing, a pre-trained production model is tracked at `weights/best.onnx` (~10.5 MB). The C++ node is configured to load this model by default, so running the training step is **optional**.

### Optional: Retraining the Model from Scratch
If you wish to modify the gate geometry, add augmentations, or retrain the network:
1. Ensure background frames are placed in `datasets/uzh-fpv-indoor-forward-davis3/img/`.
2. Generate the training dataset by projecting virtual gates onto the backgrounds:
   ```bash
   python simulation/generate_synthetic_gates.py
   ```
3. Run the training script:
   ```bash
   python simulation/train_yolo.py
   ```
   *This downloads the base YOLO weights, trains for 30 epochs, and automatically exports the new graph to `weights/best.onnx`, replacing the pre-trained default.*

---

## 5. Compiling the C++ Perception Node

Build the C++ executables using CMake:
```bash
# Go to the C++ pipeline folder
cd cpp_pipeline

# Create build directory and run CMake
mkdir -p build && cd build
cmake ..

# Compile binaries
make -j4
```
This will compile the `perception_node` executable inside `cpp_pipeline/build/`.

---

## 6. Running and Testing the Codebase

### A. Real-Time Closed-Loop UDP Simulation Test
To test the full loop-back simulation pipeline, run the following processes concurrently:

1. **Start the C++ Perception Node** (Terminal 1):
   ```bash
   ./cpp_pipeline/build/perception_node ./weights/best.onnx 127.0.0.1
   ```
2. **Start the Python Flight Simulator** (Terminal 2, from the repository root):
   ```bash
   source venv/bin/activate
   python simulation/simulate_drone.py
   ```
   *You will see the EKF state estimation prints converging in real-time in the Python terminal.*

### B. Offline Validation on Real UZH Frames
To run the trained YOLO-Pose model directly on raw images from the UZH dataset and inspect visual corner detections:
```bash
python simulation/test_on_uzh.py
```
* Bounding box corners and confidence metrics will be logged in `simulation/runs/uzh_test_results.txt`.
* Annotated verification frames will be outputted to `simulation/runs/detections/`.
