import os
import torch
from ultralytics import YOLO

def main():
    # 1. Choose device (CUDA GPU if available, else CPU)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device for training: {device}")
    if device == 0:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Load a pretrained YOLO-Pose model
    # We use the nano (n) model for fast inference and execution in C++
    model = YOLO("yolo11n-pose.pt") # downloads the weights automatically if not present

    # 3. Train the model on our synthetic gate dataset
    # We train for 30 epochs which is usually sufficient for synthetic dataset convergence
    print("Starting YOLO-Pose training...")
    results = model.train(
        data="datasets/yolo_gate.yaml",
        epochs=30,
        imgsz=320,      # Images are 346x260; 320 is close and divisible by 32 (YOLO stride)
        batch=16,
        device=device,
        project="gate_training",
        name="yolo11n_gate"
    )

    # 4. Export the trained model to ONNX format
    print("Training complete. Exporting model to ONNX format...")
    # Export with dynamic axes for flexibility in C++ OpenCV DNN or ONNX Runtime loading
    onnx_model_path = model.export(format="onnx", dynamic=True)
    print(f"Model exported successfully! ONNX model saved at: {onnx_model_path}")

if __name__ == "__main__":
    main()
