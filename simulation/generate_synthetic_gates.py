import os
import random
import glob
import numpy as np
import cv2

class SyntheticGateGenerator:
    """
    Generates synthetic training images for YOLO-Pose by projecting a 3D gate
    onto raw UZH-FPV background frames and applying physical lens distortion.
    """
    def __init__(self, bg_dir, output_dir, num_train=1000, num_val=100):
        self.bg_images = glob.glob(os.path.join(bg_dir, "*.png"))
        if not self.bg_images:
            raise RuntimeError(f"No background images found in {bg_dir}")
            
        self.output_dir = output_dir
        self.num_train = num_train
        self.num_val = num_val
        
        # Camera Intrinsics (from camchain-..yaml)
        self.img_w, self.img_h = 346, 260
        self.fx = 172.98992850734132
        self.fy = 172.98303181090185
        self.cx = 163.33639726024606
        self.cy = 134.99537889030861
        
        # Equidistant (Kannala-Brandt) Distortion Coefficients
        self.k1 = -0.027576733308582076
        self.k2 = -0.006593578674675004
        self.k3 = 0.0008566938165177085
        self.k4 = -0.00030899587045247486
        
        # Define 3D Gate geometry in its local frame (origin at center)
        # We model the gate as 1.5m wide and 1.5m high with a 0.08m (8cm) frame thickness
        self.gate_w = 1.5
        self.gate_h = 1.5
        self.gate_thickness = 0.08
        
        # Local inner corners
        w_in, h_in = self.gate_w / 2.0, self.gate_h / 2.0
        self.corners_in_local = np.array([
            [-w_in, -h_in, 0.0],  # Top-Left
            [ w_in, -h_in, 0.0],  # Top-Right
            [ w_in,  h_in, 0.0],  # Bottom-Right
            [-w_in,  h_in, 0.0]   # Bottom-Left
        ], dtype=np.float32)
        
        # Local outer corners
        w_out, h_out = w_in + self.gate_thickness, h_in + self.gate_thickness
        self.corners_out_local = np.array([
            [-w_out, -h_out, 0.0],
            [ w_out, -h_out, 0.0],
            [ w_out,  h_out, 0.0],
            [-w_out,  h_out, 0.0]
        ], dtype=np.float32)

    def get_rotation_matrix(self, roll_deg, pitch_deg, yaw_deg):
        """Computes rotation matrix from roll, pitch, yaw in degrees."""
        r = np.radians(roll_deg)
        p = np.radians(pitch_deg)
        y = np.radians(yaw_deg)
        
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])
        R_y = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])
        R_z = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])
        return R_z.dot(R_y).dot(R_x)

    def project_kannala_brandt(self, pts_3d):
        """Projects 3D points in camera frame to 2D pixels using Kannala-Brandt fisheye model."""
        pts_2d = []
        for p in pts_3d:
            X, Y, Z = p[0], p[1], p[2]
            if Z <= 0.1:
                pts_2d.append((-1.0, -1.0, 0)) # Behind camera
                continue
                
            x_norm = X / Z
            y_norm = Y / Z
            r = np.sqrt(x_norm**2 + y_norm**2)
            
            if r < 1e-6:
                # Direct projection if on optical center
                u = self.fx * x_norm + self.cx
                v = self.fy * y_norm + self.cy
                pts_2d.append((u, v, 2))
                continue
                
            theta = np.arctan(r)
            # Equidistant distortion polynomial
            theta_d = theta * (1.0 + self.k1 * (theta**2) + self.k2 * (theta**4) + 
                               self.k3 * (theta**6) + self.k4 * (theta**8))
                               
            x_distorted = (theta_d / r) * x_norm
            y_distorted = (theta_d / r) * y_norm
            
            u = self.fx * x_distorted + self.cx
            v = self.fy * y_distorted + self.cy
            
            # Check if within image bounds
            if 0 <= u < self.img_w and 0 <= v < self.img_h:
                pts_2d.append((u, v, 2))
            else:
                pts_2d.append((u, v, 1)) # Occluded / Out of bounds
        return pts_2d

    def generate_sample(self, bg_img_path):
        """Renders a gate frame on a background image and generates label."""
        bg = cv2.imread(bg_img_path, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            return None, None
        
        # 1. Randomize Gate pose in Camera frame (Z-forward, X-right, Y-down)
        # Gate is placed between 2m and 8m in front of camera
        z_trans = random.uniform(2.0, 8.0)
        # Limit horizontal/vertical translation so the gate is mostly visible
        x_trans = random.uniform(-0.5 * z_trans, 0.5 * z_trans)
        y_trans = random.uniform(-0.3 * z_trans, 0.3 * z_trans)
        
        t_cg = np.array([x_trans, y_trans, z_trans], dtype=np.float32)
        
        # Random rotation (pitch/yaw up to 35 deg, roll up to 15 deg)
        roll = random.uniform(-15, 15)
        pitch = random.uniform(-35, 35)
        yaw = random.uniform(-35, 35)
        R_cg = self.get_rotation_matrix(roll, pitch, yaw)
        
        # 2. Transform 3D corners to camera frame
        pts_in_cam = self.corners_in_local.dot(R_cg.T) + t_cg
        pts_out_cam = self.corners_out_local.dot(R_cg.T) + t_cg
        
        # 3. Project corners to 2D pixels
        corners_in_2d = self.project_kannala_brandt(pts_in_cam)
        corners_out_2d = self.project_kannala_brandt(pts_out_cam)
        
        # Ensure the inner corners are actually visible to avoid training on blank/out-of-bounds frames
        visible_in_count = sum(1 for p in corners_in_2d if p[2] == 2)
        if visible_in_count < 3:
            return None, None
            
        # 4. Render the gate onto the image
        # Create a mask for the gate frame
        mask = np.zeros_like(bg, dtype=np.uint8)
        
        # Draw the 4 quad segments of the gate frame
        for i in range(4):
            next_i = (i + 1) % 4
            # Polygon coordinates: outer[i], outer[next], inner[next], inner[i]
            poly = np.array([
                corners_out_2d[i][:2],
                corners_out_2d[next_i][:2],
                corners_in_2d[next_i][:2],
                corners_in_2d[i][:2]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
            
        # Draw a gate texture/color (in grayscale: choose random intensity 120-220)
        gate_intensity = random.randint(120, 220)
        gate_canvas = np.full_like(bg, gate_intensity)
        
        # Combine background and gate with a slight motion blur on the gate mask
        # Add random motion blur to the gate mask to simulate drone dynamics
        if random.random() > 0.4:
            blur_size = random.choice([3, 5, 7])
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
        bg_float = bg.astype(float)
        gate_float = gate_canvas.astype(float)
        mask_float = mask.astype(float) / 255.0
        
        # Alpha blending
        output_img = (bg_float * (1.0 - mask_float) + gate_float * mask_float).astype(np.uint8)
        
        # 5. Format YOLO label
        # Get bounding box around the outer corners
        pts_out_np = np.array([p[:2] for p in corners_out_2d])
        x_min, y_min = np.min(pts_out_np[:, 0]), np.min(pts_out_np[:, 1])
        x_max, y_max = np.max(pts_out_np[:, 0]), np.max(pts_out_np[:, 1])
        
        x_min_c = max(0, min(x_min, self.img_w - 1))
        y_min_c = max(0, min(y_min, self.img_h - 1))
        x_max_c = max(0, min(x_max, self.img_w - 1))
        y_max_c = max(0, min(y_max, self.img_h - 1))
        
        x_center = ((x_min_c + x_max_c) / 2.0) / self.img_w
        y_center = ((y_min_c + y_max_c) / 2.0) / self.img_h
        width = (x_max_c - x_min_c) / self.img_w
        height = (y_max_c - y_min_c) / self.img_h
        
        # YOLO-Pose keypoints line
        kp_strs = []
        for p in corners_in_2d:
            kp_x = p[0] / self.img_w
            kp_y = p[1] / self.img_h
            kp_v = p[2] # 2 = visible, 1 = occluded/out-of-bounds, 0 = behind camera
            kp_strs.append(f"{kp_x:.6f} {kp_y:.6f} {kp_v}")
            
        label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(kp_strs)
        return output_img, label_line

    def build_dataset(self):
        """Generates all samples and saves them into the YOLO directories."""
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)
            
        print("Generating synthetic dataset...")
        
        # Generate Train Set
        train_count = 0
        while train_count < self.num_train:
            bg_path = random.choice(self.bg_images)
            img, label = self.generate_sample(bg_path)
            if img is not None:
                img_name = f"synth_train_{train_count:05d}.png"
                lbl_name = f"synth_train_{train_count:05d}.txt"
                
                cv2.imwrite(os.path.join(self.output_dir, "images", "train", img_name), img)
                with open(os.path.join(self.output_dir, "labels", "train", lbl_name), "w") as f:
                    f.write(label + "\n")
                train_count += 1
                if train_count % 100 == 0:
                    print(f"  Generated {train_count}/{self.num_train} train images...")

        # Generate Val Set
        val_count = 0
        while val_count < self.num_val:
            bg_path = random.choice(self.bg_images)
            img, label = self.generate_sample(bg_path)
            if img is not None:
                img_name = f"synth_val_{val_count:05d}.png"
                lbl_name = f"synth_val_{val_count:05d}.txt"
                
                cv2.imwrite(os.path.join(self.output_dir, "images", "val", img_name), img)
                with open(os.path.join(self.output_dir, "labels", "val", lbl_name), "w") as f:
                    f.write(label + "\n")
                val_count += 1
                if val_count % 50 == 0:
                    print(f"  Generated {val_count}/{self.num_val} val images...")
                    
        print(f"Success! Generated {self.num_train} train and {self.num_val} val samples in {self.output_dir}.")

if __name__ == "__main__":
    bg_dir = "/home/aaron/Documents/perception/datasets/uzh-fpv-indoor-forward-davis3/img"
    output_dir = "/home/aaron/Documents/perception/datasets/yolo_gate"
    
    # Generate 1200 train and 150 validation samples
    generator = SyntheticGateGenerator(bg_dir, output_dir, num_train=1200, num_val=150)
    generator.build_dataset()
