import os
import numpy as np

class GateDatasetGenerator:
    """
    Generates YOLO-Pose format labels from simulator ground truth.
    Supports camera intrinsic calibration and 3D-to-2D projection.
    """
    def __init__(self, img_width=640, img_height=480, fx=400, fy=400, cx=320, cy=240):
        self.img_width = img_width
        self.img_height = img_height
        
        # Camera Intrinsic Matrix (K)
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float32)
        
        # Camera to Body Frame extrinsics (Default: Camera is 5cm forward from drone center)
        self.R_CB = np.eye(3, dtype=np.float32)  # Assumed aligned for simplicity
        self.t_CB = np.array([0.05, 0.0, 0.0], dtype=np.float32) 

    def project_3d_to_2d(self, p_W_corners, p_W_drone, R_W_drone):
        """
        Projects 3D gate corners in World frame to 2D image coordinates.
        
        Parameters:
        - p_W_corners: numpy array of shape (4, 3) representing 3D coordinates of gate corners.
        - p_W_drone: numpy array of shape (3,) representing drone position.
        - R_W_drone: numpy array of shape (3, 3) representing drone rotation matrix (World to Body).
        """
        # Rotation matrix from World to Body frame: R_BW = R_W_drone.T
        R_BW = R_W_drone.T
        
        corners_2d = []
        for p_W_corner in p_W_corners:
            # 1. Translate corner relative to drone: p_B = R_BW * (p_W_corner - p_W_drone)
            p_B = R_BW.dot(p_W_corner - p_W_drone)
            
            # 2. Transform from Body to Camera frame: p_C = R_CB * p_B + t_CB
            p_C = self.R_CB.dot(p_B) + self.t_CB
            
            # Check if point is behind the camera
            if p_C[2] <= 0.1:
                corners_2d.append((0.0, 0.0, 0)) # Not visible / behind camera
                continue
                
            # 3. Project to image plane
            p_img = self.K.dot(p_C) / p_C[2]
            u, v = p_img[0], p_img[1]
            
            # Check boundaries
            if 0 <= u < self.img_width and 0 <= v < self.img_height:
                corners_2d.append((u, v, 2))  # Visible
            else:
                corners_2d.append((u, v, 1))  # Occluded / Out of bounds
                
        return corners_2d

    def generate_yolo_pose_label(self, corners_2d):
        """
        Formats 2D corners into a single YOLO-Pose label string.
        YOLO format: class_idx x_center y_center width height kp1_x kp1_y kp1_v ...
        All coordinates are normalized between 0 and 1.
        """
        vis_corners = [pt for pt in corners_2d if pt[2] == 2]
        if len(vis_corners) < 2:
            return None # Gate not visible enough to detect or label
            
        pts = np.array([(pt[0], pt[1]) for pt in corners_2d])
        
        # Calculate bounding box from keypoints
        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
        
        # Clamp bbox coordinates to image boundaries for YOLO format
        x_min_clamp = max(0.0, min(x_min, self.img_width - 1))
        y_min_clamp = max(0.0, min(y_min, self.img_height - 1))
        x_max_clamp = max(0.0, min(x_max, self.img_width - 1))
        y_max_clamp = max(0.0, min(y_max, self.img_height - 1))
        
        x_center = (x_min_clamp + x_max_clamp) / 2.0 / self.img_width
        y_center = (y_min_clamp + y_max_clamp) / 2.0 / self.img_height
        width = (x_max_clamp - x_min_clamp) / self.img_width
        height = (y_max_clamp - y_min_clamp) / self.img_height
        
        # Format keypoints
        kp_str = []
        for pt in corners_2d:
            kp_x = pt[0] / self.img_width
            kp_y = pt[1] / self.img_height
            kp_v = pt[2] # 0 = invisible, 1 = occluded/out of bounds, 2 = visible
            kp_str.append(f"{kp_x:.6f} {kp_y:.6f} {kp_v}")
            
        label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(kp_str)
        return label_line

# Example usage/validation
if __name__ == "__main__":
    generator = GateDatasetGenerator()
    
    # 3D Gate Corners in World Frame (e.g. 1.5m x 1.5m gate centered at [5.0, 0.0, 1.0])
    gate_corners = np.array([
        [5.0, -0.75,  1.75],  # Top Left
        [5.0,  0.75,  1.75],  # Top Right
        [5.0,  0.75,  0.25],  # Bottom Right
        [5.0, -0.75,  0.25]   # Bottom Left
    ])
    
    # Drone Pose in World Frame (e.g. drone at [2.0, 0.1, 1.0] facing straight ahead)
    drone_pos = np.array([2.0, 0.1, 1.0])
    drone_rot = np.eye(3) # No rotation
    
    corners_2d = generator.project_3d_to_2d(gate_corners, drone_pos, drone_rot)
    print("Projected 2D Corners (u, v, visibility):")
    for idx, corner in enumerate(corners_2d):
        print(f"  Corner {idx+1}: u={corner[0]:.2f}, v={corner[1]:.2f}, vis={corner[2]}")
        
    yolo_line = generator.generate_yolo_pose_label(corners_2d)
    print("\nGenerated YOLO-Pose Label Line:")
    print(yolo_line)
