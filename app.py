"""
app.py  –  2D to 3D Reconstruction
Run:  python app.py
Open in Chrome/Firefox with WebGL enabled for interactive 3D view.

Tabs:
  🖥️  CPU Mode  – Fast depth-map mesh (DepthAnything V2, no GPU needed)
  🎮  GPU Mode  – Full 360° mesh (TripoSR, requires ~6 GB VRAM)

Install TripoSR on the GPU machine:
  pip install git+https://github.com/VAST-AI-Research/TripoSR
  pip install rembg
"""

import os
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

import trimesh
import gradio as gr

# ── TripoSR optional import ────────────────────────────────────────────────────
TRIPOSR_AVAILABLE = False
try:
    import torch
    import rembg
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
    TRIPOSR_AVAILABLE = True
    print("TripoSR detected ✓")
except ImportError:
    print("TripoSR not installed – GPU Mode will show an error if used.")
    print("  → pip install git+https://github.com/VAST-AI-Research/TripoSR")

# ── Load depth model once at startup (CPU) ─────────────────────────────────────
print("Loading DepthAnything V2 …")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
print("DepthAnything V2 ready.")

OUTPUT_DIR = "gradio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── TripoSR lazy loader ────────────────────────────────────────────────────────
_triposr_model  = None
_rembg_session  = None

def _get_triposr():
    """Load TripoSR + rembg once on first call; reuse afterwards."""
    global _triposr_model, _rembg_session
    if _triposr_model is None:
        if not TRIPOSR_AVAILABLE:
            raise RuntimeError("TripoSR is not installed on this machine.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading TripoSR on {device} (first-time, may take a moment)…")
        _triposr_model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        _triposr_model.renderer.set_chunk_size(8192)
        _triposr_model.to(device)
        _rembg_session = rembg.new_session()
        print("TripoSR ready.")
    return _triposr_model, _rembg_session


def _gpu_status():
    """Return a human-readable GPU / TripoSR status string."""
    if not TRIPOSR_AVAILABLE:
        return (
            "❌ **TripoSR not installed on this machine.**  \n"
            "Run on the GPU machine:  \n"
            "`pip install git+https://github.com/VAST-AI-Research/TripoSR rembg`"
        )
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"✅ **GPU detected:** {name} ({vram:.1f} GB VRAM) — ready to generate"
    return (
        "⚠️ **No GPU detected.** TripoSR will run on CPU (~5-10 min per model).  \n"
        "For best performance run this on a machine with a CUDA GPU."
    )


# ── CPU Pipeline ───────────────────────────────────────────────────────────────
def reconstruct(pil_image):
    if pil_image is None:
        return None

    print("Estimating depth…")
    result   = depth_pipe(pil_image)
    depth_np = np.array(result["depth"])
    img_np   = np.array(pil_image)

    h, w = img_np.shape[:2]
    depth_np = cv2.resize(depth_np, (w, h))

    factor = 4
    wv = w // factor
    hv = h // factor

    xx, yy = np.meshgrid(np.linspace(0, 1, wv), np.linspace(0, 1, hv))

    img_flipped = cv2.flip(img_np, 1)
    img_small   = cv2.resize(img_flipped, (wv, hv))
    colors_rgb  = img_small.reshape(-1, 3)
    colors_rgba = np.column_stack(
        [colors_rgb, np.full(len(colors_rgb), 255, dtype=np.uint8)]
    )

    xi = np.round(xx * (w - 1)).astype(int).clip(0, w - 1)
    yi = np.round(yy * (h - 1)).astype(int).clip(0, h - 1)
    z  = depth_np[yi, xi]

    z_low, z_high = np.percentile(z, 2), np.percentile(z, 98)
    z = np.clip(z, z_low, z_high)
    z = (z - z_low) / (z_high - z_low + 1e-8)
    z = z * 0.5

    verts = np.column_stack(
        [(xx.flatten() - 0.5) * -1, (yy.flatten() - 0.5) * -1, z.flatten()]
    )

    r, c = np.meshgrid(np.arange(hv - 1), np.arange(wv - 1), indexing="ij")
    v1 = (r * wv + c).flatten()
    v2 = v1 + 1
    v3 = v1 + wv
    v4 = v1 + wv + 1
    faces = np.vstack([
        np.column_stack([v1, v2, v3]),
        np.column_stack([v2, v4, v3]),
    ])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=colors_rgba,
        process=False,
    )
    mat = trimesh.visual.material.PBRMaterial(doubleSided=True)
    mesh.visual.material = mat

    glb_path = os.path.join(OUTPUT_DIR, "model.glb")
    mesh.export(glb_path)
    print(f"Saved: {glb_path}")
    return glb_path


# ── Statistical Outlier Removal (CPU) ─────────────────────────────────────────
def apply_sor(nb_neighbors=20, std_ratio=2.0):
    """Load the last CPU model, apply SOR on vertices, re-export."""
    glb_path = os.path.join(OUTPUT_DIR, "model.glb")
    if not os.path.exists(glb_path):
        print("No CPU model found. Generate one first.")
        return None

    print(f"Applying SOR (k={nb_neighbors}, std={std_ratio})…")
    mesh  = trimesh.load(glb_path, force="mesh")
    verts = np.array(mesh.vertices, dtype=np.float32)
    n     = len(verts)
    k     = min(nb_neighbors + 1, n)

    BATCH      = 64
    mean_dists = np.empty(n, dtype=np.float32)
    for start in range(0, n, BATCH):
        end    = min(start + BATCH, n)
        diff   = verts[np.newaxis, :, :] - verts[start:end, np.newaxis, :]
        dists  = np.sqrt((diff * diff).sum(axis=2))
        nearest = np.partition(dists, k, axis=1)[:, 1:k]
        mean_dists[start:end] = nearest.mean(axis=1)
    print(f"  k-NN computed for {n} vertices")

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    keep_mask = mean_dists <= threshold
    keep_idx  = np.where(keep_mask)[0]
    print(f"SOR removed {(~keep_mask).sum()} / {n} vertices")

    old_to_new = np.full(n, -1, dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx))

    faces     = np.array(mesh.faces)
    face_mask = keep_mask[faces].all(axis=1)
    new_faces = old_to_new[faces[face_mask]]
    new_verts = verts[keep_idx]
    new_colors = np.array(mesh.visual.vertex_colors)[keep_idx] \
        if mesh.visual.kind == "vertex" else None

    clean_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=new_faces,
        vertex_colors=new_colors, process=False,
    )
    clean_mesh.visual.material = trimesh.visual.material.PBRMaterial(doubleSided=True)

    sor_path = os.path.join(OUTPUT_DIR, "model_sor.glb")
    clean_mesh.export(sor_path)
    print(f"Saved SOR model: {sor_path}")
    return sor_path


# ── GPU Pipeline (TripoSR 360°) ────────────────────────────────────────────────
def triposr_preprocess(pil_image, do_remove_bg, foreground_ratio):
    """Remove background + normalize image for TripoSR input."""
    if pil_image is None:
        raise gr.Error("Please upload an image first.")
    if not TRIPOSR_AVAILABLE:
        raise gr.Error(
            "TripoSR is not installed. Run on the GPU machine:\n"
            "pip install git+https://github.com/VAST-AI-Research/TripoSR rembg"
        )

    _, rembg_sess = _get_triposr()

    def fill_bg(image):
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    if do_remove_bg:
        image = pil_image.convert("RGB")
        image = remove_background(image, rembg_sess)
        image = resize_foreground(image, foreground_ratio)
        image = fill_bg(image)
    else:
        image = pil_image
        if image.mode == "RGBA":
            image = fill_bg(image)

    print("Preprocessing done.")
    return image


def triposr_generate(preprocessed_image, mc_resolution):
    """Run TripoSR inference → export full 360° GLB."""
    if preprocessed_image is None:
        raise gr.Error("Click '1. Preprocess Image' first.")
    if not TRIPOSR_AVAILABLE:
        raise gr.Error(
            "TripoSR is not installed. Run on the GPU machine:\n"
            "pip install git+https://github.com/VAST-AI-Research/TripoSR rembg"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, _ = _get_triposr()

    print(f"Running TripoSR on {device}…")
    with torch.no_grad():
        scene_codes = model(preprocessed_image, device=device)

    mesh = model.extract_mesh(scene_codes, True, resolution=int(mc_resolution))[0]
    mesh = to_gradio_3d_orientation(mesh)

    glb_path = os.path.join(OUTPUT_DIR, "model_360.glb")
    mesh.export(glb_path)
    print(f"TripoSR 360° model saved: {glb_path}")
    return glb_path


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="2D to 3D Reconstruction",
    theme=gr.themes.Base(),
) as demo:

    gr.Markdown("<h1 style='text-align: center;'>2D → 3D Reconstruction</h1>")
    gr.Markdown("<p style='text-align: center;'>Upload any 2D image and generate a textured 3D mesh.</p>")

    with gr.Tabs():

        # ── Tab 1: CPU Mode ────────────────────────────────────────────────────
        with gr.Tab("🖥️ CPU Mode — Fast Preview"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Image", height=320)
                    btn       = gr.Button("Generate 3D Model", variant="primary")
                    sor_btn   = gr.Button("Apply SOR (Remove Outliers)", variant="secondary")
                with gr.Column(scale=2):
                    model_out = gr.Model3D(
                        label="Interactive 3D View (drag to rotate)",
                        clear_color=[1.0, 1.0, 1.0, 1.0],
                        height=500,
                    )

            btn.click(fn=reconstruct, inputs=img_input, outputs=model_out)
            sor_btn.click(fn=apply_sor, inputs=[], outputs=model_out)

        # ── Tab 2: GPU Mode (TripoSR 360°) ────────────────────────────────────
        with gr.Tab("🎮 GPU Mode — 360° TripoSR"):
            with gr.Row():
                with gr.Column(scale=1):
                    gpu_img_input = gr.Image(
                        type="pil", label="Upload Image",
                        image_mode="RGBA", height=260,
                    )
                    with gr.Group():
                        remove_bg_chk = gr.Checkbox(
                            label="Remove Background (recommended)", value=True
                        )
                        fg_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                        )
                        mc_res = gr.Slider(
                            label="Mesh Resolution (higher = more VRAM)",
                            minimum=32, maximum=320, value=256, step=32,
                        )
                    preprocess_btn = gr.Button("1. Preprocess Image", variant="secondary")
                    generate_btn   = gr.Button("2. Generate 360° Model", variant="primary")

                with gr.Column(scale=2):
                    preprocessed_img = gr.Image(
                        label="Preprocessed (input to TripoSR)",
                        interactive=False, height=220,
                    )
                    gpu_model_out = gr.Model3D(
                        label="360° Interactive 3D View (drag to rotate)",
                        clear_color=[0.9, 0.9, 0.9, 1.0],
                        height=420,
                    )

            preprocess_btn.click(
                fn=triposr_preprocess,
                inputs=[gpu_img_input, remove_bg_chk, fg_ratio],
                outputs=preprocessed_img,
            )
            generate_btn.click(
                fn=triposr_generate,
                inputs=[preprocessed_img, mc_res],
                outputs=gpu_model_out,
            )


if __name__ == "__main__":
    demo.launch(share=True)
