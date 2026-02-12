import gradio as gr
import os
import shutil
import numpy as np
from PIL import Image
import open3d as o3d
from transformers import pipeline
import cv2
import trimesh

# --- Define Pipeline Functions ---

def run_depth_anything(image):
    """
    Run DepthAnything V2 on the input image.
    Returns: depth map (numpy array)
    """
    print("Estimating depth...")
    try:
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        depth_result = pipe(image)
        depth_map = np.array(depth_result["depth"])
        return depth_map
    except Exception as e:
        print(f"Depth error: {e}")
        return None

def generate_textured_glb(image, depth_map, output_dir):
    """
    Generate a GLB file (Texture Embedded) designed to be BRIGHT (Unlit).
    """
    print("Generating GLB mesh...")
    img = np.array(image)
    h, w = img.shape[:2]
    
    # Resize depth to match
    depth_h, depth_w = depth_map.shape[:2]
    if (depth_w, depth_h) != (w, h):
        depth_map = cv2.resize(depth_map, (w, h))

    # Downsample (keep it light)
    downsample_factor = 4
    w_vert = w // downsample_factor
    h_vert = h // downsample_factor
    
    # Grid
    xx, yy = np.meshgrid(np.linspace(0, w-1, w_vert), np.linspace(0, h-1, h_vert))
    x_idx = np.round(xx).astype(int).clip(0, w-1)
    y_idx = np.round(yy).astype(int).clip(0, h-1)
    
    z_grid = depth_map[y_idx, x_idx]
    z_norm = (z_grid - z_grid.min()) / (z_grid.max() - z_grid.min())
    z_flat = z_norm.flatten() * 500.0 # Scale
    
    vertices = np.column_stack((xx.flatten(), yy.flatten(), z_flat))

    # Triangles
    rows, cols = np.meshgrid(np.arange(h_vert-1), np.arange(w_vert-1), indexing='ij')
    v1 = rows * w_vert + cols
    v2 = v1 + 1
    v3 = v1 + w_vert
    v4 = v1 + w_vert + 1
    
    t1 = np.column_stack((v1.flatten(), v2.flatten(), v3.flatten()))
    t2 = np.column_stack((v2.flatten(), v4.flatten(), v3.flatten()))
    faces = np.vstack((t1, t2))

    # UVs
    u = xx.flatten() / (w - 1)
    v = 1.0 - (yy.flatten() / (h - 1))
    uvs = np.column_stack((u, v))

    # Build Trimesh Object
    print("Packing texture into GLB using Trimesh...")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Use PBR Material with 'baseColorTexture' to prevent it being black
    # Making it emissive or unlit often helps in web viewers
    material = trimesh.visual.material.PBRMaterial(
        name='material_0',
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=Image.fromarray(img),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True
    )
    
    # Create Visuals
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    # Correct Orientation attempt 2
    # If it was upside down, we rotate 180 around Z to flip it upright.
    rot = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
    mesh.apply_transform(rot)

    # Save as GLB
    glb_path = os.path.join(output_dir, "model.glb")
    mesh.export(glb_path)
    
    print(f"GLB Saved to {glb_path}")
    return glb_path

def process_image(input_image):
    if input_image is None:
        return None
    
    output_dir = "gradio_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    depth_map = run_depth_anything(input_image)
    if depth_map is None:
        return None
        
    glb_path = generate_textured_glb(input_image, depth_map, output_dir)
    
    return glb_path


# --- CUSTOM CSS (Dark & Premium) ---
custom_css = """
body, .gradio-container {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
}
h1 {
    text-align: center;
    color: #f1f5f9;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 10px;
    padding: 20px 0;
}
#desc {
    text-align: center;
    font-size: 1.1rem;
    color: #cbd5e1;
    margin-bottom: 30px;
}
"""

# --- UI Layout ---
# Using a Soft theme
theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="stone",
    neutral_hue="stone",
)

with gr.Blocks(title="2D to 3D", css=custom_css, theme=theme) as app:
    
    with gr.Column():
        gr.Markdown("# 2D to 3D Reconstruction Model")
        gr.Markdown(
            "Upload a 2D image to generate a high-fidelity **Textured 3D Mesh**.",
            elem_id="desc"
        )
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            input_img = gr.Image(
                type="pil", 
                label="Input Image", 
                height=350,
                sources=['upload', 'clipboard'],
                elem_classes="dark-panel"
            )
            run_btn = gr.Button("✨ Generate 3D Model", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # 3D Viewer with Dark Background (Cinematic)
            output_3d = gr.Model3D(
                label="Interactive 3D Preview", 
                clear_color=[0.05, 0.05, 0.1, 1.0], # Dark Navy/Black background
                interactive=True,
                height=600,
                elem_classes="dark-panel"
            )
            download_file = gr.File(label="Download .GLB Asset")

    def wrapper(img):
        result_path = process_image(img)
        return result_path, result_path 

    run_btn.click(
        fn=wrapper,
        inputs=[input_img],
        outputs=[output_3d, download_file]
    )

if __name__ == "__main__":
    app.launch()
