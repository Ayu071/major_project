import open3d as o3d
import numpy as np
import os
from transformers import pipeline
import cv2
from PIL import Image
import argparse
import copy

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img), img

def get_depth_map(pipe, pil_image):
    result = pipe(pil_image)
    depth = np.array(result["depth"])
    return depth

def create_pcd_from_depth(rgb_image, depth_map, output_file=None):
    # Resize depth to match image
    h, w = rgb_image.shape[:2]
    d_h, d_w = depth_map.shape[:2]
    if (d_w, d_h) != (w, h):
        depth_map = cv2.resize(depth_map, (w, h))

    # Open3D structures
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_map.astype(np.float32))
    
    # Intrinsic (focal length)
    focal_length = w * 0.8
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, focal_length, focal_length, w/2, h/2)
    
    # Generate RGBD
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
    )
    
    # Generate Point Cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # Flip to be upright
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd

def multi_view_fusion(image_folder, num_views, output_path="output/merged_model.ply"):
    """
    Stitches multiple images into a single 3D point cloud assuming a turntable rotation.
    """
    print(f"--- Multi-View Fusion (Turntable Mode) ---")
    
    # 1. Load Model
    print("Loading DepthAnything V2...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    
    # 2. Get Images
    # We look for all .jpg/.png in folder and sort them
    files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    
    # Limit to num_views if specified
    if num_views > 0:
        files = files[:num_views]
    
    if len(files) < 2:
        print("Error: Need at least 2 images for fusion.")
        return

    print(f"Processing {len(files)} views...")
    
    combined_pcd = o3d.geometry.PointCloud()
    
    # 3. Process Each View
    # We assume the object rotates 360 degrees over the sequence
    angle_step = 360.0 / len(files)
    
    for i, filename in enumerate(files):
        img_path = os.path.join(image_folder, filename)
        print(f"  View {i+1}/{len(files)}: {filename}")
        
        # A. Depth Estimation
        pil_img, cv_img = preprocess_image(img_path)
        depth_map = get_depth_map(pipe, pil_img)
        
        # B. Make Point Cloud
        pcd = create_pcd_from_depth(cv_img, depth_map)
        
        # C. Rotate (Align)
        # If object rotates on turntable, effectively the CAMERA rotates around it.
        # We rotate the point cloud back to a common frame.
        # Rotate around Y axis (standard Up)
        rotation_angle = np.radians(angle_step * i)
        
        # Create rotation matrix (Y-axis rotation)
        # [ cos  0  sin ]
        # [  0   1   0  ]
        # [ -sin 0  cos ]
        R = pcd.get_rotation_matrix_from_xyz((0, rotation_angle, 0))
        
        # We rotate the cloud. 
        # Note: Center of rotation is crucial. We assume object is centered at (0,0,Z).
        # Depth maps usually start at Z=0. We need to center the object first.
        center = pcd.get_center()
        pcd.translate(-center) # Move to origin
        pcd.rotate(R, center=(0,0,0)) # Rotate
        pcd.translate(center) # Move back? Actually for fusion we usually keep at origin.
        
        # Refinement: ICP (Iterative Closest Point)
        # If this isn't the first cloud, align it to the previous/combined one for tight fit
        if i > 0:
            print("    Aligning to previous view...")
            # Use point-to-point ICP for fine tuning
            threshold = 0.02
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd, combined_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            pcd.transform(reg_p2p.transformation)
        
        # Add to main cloud
        combined_pcd += pcd

    # 4. Final Cleanup
    print("Merging and cleaning...")
    # Downsample to remove duplicate points from overlaps
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
    
    # Statistical Outlier Removal (Clean noise)
    cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    combined_pcd = combined_pcd.select_by_index(ind)

    # 5. Save
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print(f"✓ Saved merged model to: {output_path}")
    
    # Visualize
    o3d.visualization.draw_geometries([combined_pcd], window_name="Merged 3D Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing image sequence")
    parser.add_argument("--views", type=int, default=0, help="Number of views to use (0=all)")
    parser.add_argument("--output", type=str, default="output/merged_model.ply")
    
    args = parser.parse_args()
    multi_view_fusion(args.folder, args.views, args.output)
