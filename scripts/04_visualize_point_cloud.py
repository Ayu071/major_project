import open3d as o3d
import os
import sys

def visualize_point_cloud(ply_path):
    if not os.path.exists(ply_path):
        print(f"Error: Point cloud file not found at {ply_path}")
        return
    
    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    num_points = len(pcd.points)
    print(f"✓ Loaded {num_points:,} points")
    print("\nVisualization Controls:")
    print("  - Left-click + drag: Rotate")
    print("  - Right-click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Close window to exit")
    print("\nOpening visualization window...")
    
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {os.path.basename(ply_path)}", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set a nice initial view
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()
    print("Visualization closed.")

if __name__ == "__main__":
    # Check for command-line argument
    if len(sys.argv) > 1:
        ply_path = sys.argv[1]
    else:
        # Default to RGB point cloud
        ply_path = os.path.join("output", "point_cloud_rgb.ply")
    
    visualize_point_cloud(ply_path)
