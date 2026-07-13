import os
import glob
import cv2
from ultralytics import YOLO

def main():
    # 1. Load the trained model (we can use the PyTorch weights for easy python execution)
    model_path = "weights/best.pt"
    if not os.path.exists(model_path):
        # Fallback to ONNX if .pt is missing
        model_path = "weights/best.onnx"
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 2. Get UZH-FPV raw images
    uzh_img_dir = "datasets/uzh-fpv-indoor-forward-davis3/img"
    images = sorted(glob.glob(os.path.join(uzh_img_dir, "*.png")))
    print(f"Found {len(images)} raw images in the UZH-FPV dataset.")

    # 3. Create output directory for detections
    output_detect_dir = "simulation/runs/detections"
    os.makedirs(output_detect_dir, exist_ok=True)
    
    # 4. Prepare log file
    log_file_path = "simulation/runs/uzh_test_results.txt"
    log_entries = []

    # 5. Run inference on a sample of images
    # We sample every 20th frame to cover a wide part of the flight trajectory
    sample_rate = 20
    test_images = images[::sample_rate]
    print(f"Testing model on {len(test_images)} sampled frames...")

    detected_count = 0
    for idx, img_path in enumerate(test_images):
        img_name = os.path.basename(img_path)
        
        # Run inference (confidence threshold = 0.5)
        results = model.predict(img_path, conf=0.5, verbose=False)
        result = results[0]
        
        num_detections = len(result.boxes)
        if num_detections > 0:
            detected_count += 1
            log_entries.append(f"Frame: {img_name} | Detections: {num_detections}")
            
            # Extract bounding boxes and keypoints
            for i, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].tolist() # Bounding box coords
                conf = box.conf[0].item()    # Confidence score
                
                log_entry = f"  Gate {i+1}: Bbox={xyxy}, Conf={conf:.4f}"
                print(f"  [Detection] {img_name}: Gate {i+1} found with confidence {conf:.4f}")
                
                # Extract keypoints (4 gate corners)
                if result.keypoints is not None:
                    kpts = result.keypoints.xy[i].tolist() # Keypoints for the i-th box
                    log_entry += f", Corners={kpts}"
                log_entries.append(log_entry)
            
            # Save the annotated frame to disk
            annotated_img = result.plot(labels=True, boxes=True, conf=True)
            cv2.imwrite(os.path.join(output_detect_dir, f"detect_{img_name}"), annotated_img)
            
    # Write logs to file
    with open(log_file_path, "w") as f:
        f.write(f"UZH-FPV Offline Inference Verification Report\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Total frames tested: {len(test_images)}\n")
        f.write(f"Frames with gate detections (conf > 0.5): {detected_count}\n")
        f.write("="*50 + "\n")
        f.write("\n".join(log_entries) + "\n")
        
    print(f"\nVerification complete!")
    print(f"Saved detection report to: {log_file_path}")
    print(f"Saved {detected_count} annotated images with gate overlays to: {output_detect_dir}")

if __name__ == "__main__":
    main()
