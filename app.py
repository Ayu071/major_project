"""
app.py  –  2D to 3D Reconstruction
Run:  python app.py
Open in Chrome/Firefox with WebGL enabled for interactive 3D view.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import trimesh
import gradio as gr

# ── Load depth model once at startup ──────────────────────────────────────────
print("Loading DepthAnything V2 …")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
print("Model ready.")

OUTPUT_DIR = "gradio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Core function ──────────────────────────────────────────────────────────────
def reconstruct(pil_image):
    if pil_image is None:
        return None

    print("Estimating depth...")
    result   = depth_pipe(pil_image)
    depth_np = np.array(result["depth"])
    img_np   = np.array(pil_image)

    h, w = img_np.shape[:2]

    # Resize depth to match image
    depth_np = cv2.resize(depth_np, (w, h))

    # Downsample for mesh (factor 4)
    factor = 4
    wv = w // factor
    hv = h // factor

    # Grid (0 to 1)
    xx, yy = np.meshgrid(np.linspace(0, 1, wv), np.linspace(0, 1, hv))

    # Vertex Colors
    # Pre-flip image horizontally to cancel the X-axis geometry flip below
    img_flipped = cv2.flip(img_np, 1)
    img_small   = cv2.resize(img_flipped, (wv, hv))
    colors_rgb  = img_small.reshape(-1, 3)
    colors_rgba = np.column_stack(
        [colors_rgb, np.full(len(colors_rgb), 255, dtype=np.uint8)]
    )

    # Depth values
    xi = np.round(xx * (w - 1)).astype(int).clip(0, w - 1)
    yi = np.round(yy * (h - 1)).astype(int).clip(0, h - 1)
    z  = depth_np[yi, xi]

    # Normalize Z (no inversion — DepthAnything higher = closer = protrudes out)
    z  = (z - z.min()) / (z.max() - z.min() + 1e-8)
    z  = z * 0.5

    # Vertices: X flipped so mesh faces camera correctly
    verts = np.column_stack(
        [(xx.flatten() - 0.5) * -1, (yy.flatten() - 0.5) * -1, z.flatten()]
    )

    # Faces (two triangles per quad)
    r, c = np.meshgrid(np.arange(hv - 1), np.arange(wv - 1), indexing="ij")
    v1 = (r * wv + c).flatten()
    v2 = v1 + 1
    v3 = v1 + wv
    v4 = v1 + wv + 1
    faces = np.vstack([
        np.column_stack([v1, v2, v3]),
        np.column_stack([v2, v4, v3]),
    ])

    # Build mesh with vertex colors
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=colors_rgba,
        process=False,
    )

    # Force double-sided rendering via GLB material
    mat = trimesh.visual.material.PBRMaterial(doubleSided=True)
    mesh.visual.material = mat

    glb_path = os.path.join(OUTPUT_DIR, "model.glb")
    mesh.export(glb_path)
    print(f"Saved: {glb_path}")
    return glb_path


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="2D to 3D Reconstruction",
    theme=gr.themes.Base(),
) as demo:

    gr.Markdown("<h1 style='text-align: center;'>2D to 3D Reconstruction</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload any 2D image to generate a textured 3D mesh reconstruction.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Image", height=320)
            btn = gr.Button("Generate 3D Model", variant="primary")

        with gr.Column(scale=2):
            model_out = gr.Model3D(
                label="Interactive 3D View (drag to rotate)",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                height=500,
            )

    btn.click(fn=reconstruct, inputs=img_input, outputs=model_out)

if __name__ == "__main__":
    demo.launch(share=True)
