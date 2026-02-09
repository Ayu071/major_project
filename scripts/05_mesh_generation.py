import open3d as o3d
import numpy as np
import os
import argparse

def create_mesh_from_points(pcd_path, output_path, depth=8, density_threshold=0.1, visualize=True):
    """
    Step 5: Basic Mesh Generation (Geometry Only)
    
    This script converts the Point Cloud into a solid mesh using Poisson Reconstruction.
    It focuses on GEOMETRY, while Step 6 focuses on TEXTURE.
    """
    
    if not os.path.exists(pcd_path):
        print(f"Error: File not found at {pcd_path}")
        return

    print(f"Loading Point Cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Downsample if too large
    if len(pcd.points) > 500000:
        print("Downsampling for performance...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Orientation
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Poisson Reconstruction
    print(f"Running Poisson Reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    
    # Cleanup
    print("Cleaning mesh artifacts...")
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Save
    print(f"Saving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    print("✓ Mesh saved")

    if visualize:
        print("Visualizing...")
        o3d.visualization.draw_geometries([mesh], window_name="Step 5: Geometry Mesh", mesh_show_back_face=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join("output", "point_cloud_rgb.ply"))
    parser.add_argument("--output", default=os.path.join("output", "model_mesh_geometry.obj"))
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    create_mesh_from_points(args.input, args.output, args.depth, visualize=args.visualize)
