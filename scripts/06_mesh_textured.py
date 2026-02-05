import open3d as o3d
import numpy as np
from PIL import Image
import os
import argparse
import shutil

def create_lightweight_textured_mesh(image_path, depth_map_path, output_path, depth_scale=500.0, downsample_factor=5):
    """
    Creates a Resource-Friendly Textured Mesh and packages it for viewing.
    """
    
    print(f"--- Low System Resource Mode ---")
    
    if not os.path.exists(depth_map_path) or not os.path.exists(image_path):
        print(f"Error: Missing input files.")
        return
    
    # 1. Load Data
    depth_map = np.load(depth_map_path)
    pil_img = Image.open(image_path)
    img_w, img_h = pil_img.size
    
    # Resize depth
    dh, dw = depth_map.shape[:2]
    if (dw, dh) != (img_w, img_h):
        depth_pil = Image.fromarray(depth_map)
        depth_pil = depth_pil.resize((img_w, img_h), Image.NEAREST)
        depth_map = np.array(depth_pil)

    # 2. Downsample Geometry
    w_vert = img_w // downsample_factor
    h_vert = img_h // downsample_factor
    
    print(f"Generating Mesh ({w_vert}x{h_vert})...")
    
    xx, yy = np.meshgrid(np.linspace(0, img_w-1, w_vert), np.linspace(0, img_h-1, h_vert))
    x_idx = np.round(xx).astype(int).clip(0, img_w-1)
    y_idx = np.round(yy).astype(int).clip(0, img_h-1)
    
    z_grid = depth_map[y_idx, x_idx]
    z_norm = (z_grid - z_grid.min()) / (z_grid.max() - z_grid.min())
    z_flat = z_norm.flatten() * depth_scale
    
    vertices = np.column_stack((xx.flatten(), yy.flatten(), z_flat))

    # 3. Triangles
    rows, cols = np.meshgrid(np.arange(h_vert-1), np.arange(w_vert-1), indexing='ij')
    v1 = rows * w_vert + cols
    v2 = v1 + 1
    v3 = v1 + w_vert
    v4 = v1 + w_vert + 1
    
    t1 = np.column_stack((v1.flatten(), v2.flatten(), v3.flatten()))
    t2 = np.column_stack((v2.flatten(), v4.flatten(), v3.flatten()))
    triangles = np.vstack((t1, t2))

    # 4. UVs
    u = xx.flatten() / (img_w - 1)
    v = 1.0 - (yy.flatten() / (img_h - 1))
    uvs = np.column_stack((u, v))
    tri_uvs = uvs[triangles.flatten()]

    # 5. Build Mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
    
    # Transform
    mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 6. Save OBJ
    print(f"Saving to {output_path}...")
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    # 7. Auto-Package for Viewing (Fix Material)
    package_dir = "final_model"
    if not os.path.exists(package_dir):
        os.makedirs(package_dir)
        
    img_filename = os.path.basename(image_path)
    
    # Write .mtl
    mtl_content = f"""newmtl skin
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
d 1.0
illum 1
map_Kd {img_filename}
"""
    with open(os.path.join(package_dir, "model.mtl"), "w") as f:
        f.write(mtl_content)
        
    # Copy Image
    shutil.copy(image_path, os.path.join(package_dir, img_filename))
    
    # Fix OBJ and save to package
    dst_obj = os.path.join(package_dir, "model.obj")
    
    # We read the raw OBJ we just saved, and prepend the material link
    with open(output_path, "r") as f_in:
        content = f_in.read()
        
    with open(dst_obj, "w") as f_out:
        f_out.write("mtllib model.mtl\n")
        f_out.write("usemtl skin\n")
        f_out.write(content)
        
    print(f"✓ Package ready in: {package_dir}")

    # Visualize
    print("\nVisualizing...")
    mesh.textures = [o3d.geometry.Image(np.asarray(pil_img))]
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Textured 3D Mesh")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=os.path.join("assets", "demo04.jpg"))
    parser.add_argument("--depth", default=os.path.join("output", "depth_map.npy"))
    parser.add_argument("--output", default=os.path.join("output", "textured_mesh.obj"))
    parser.add_argument("--scale", type=float, default=500.0)
    parser.add_argument("--quality", type=int, default=5)
    
    args = parser.parse_args()
    create_lightweight_textured_mesh(args.image, args.depth, args.output, args.scale, args.quality)
