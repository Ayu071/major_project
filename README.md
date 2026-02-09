# 2D to 3D Pipeline: DepthAnything V2

This project implements a complete pipeline to convert single 2D images into 3D Point Clouds and Textured Meshes using State-of-the-Art Monocular Depth Estimation.

## 🚀 Overview

The pipeline consists of 6 sequential steps:
1.  **Image Basics**: Load and inspect input images.
2.  **Preprocessing**: Resize and prepare images for AI.
3.  **Depth Estimation**: Generate high-quality depth maps using `DepthAnything-V2`.
4.  **Point Cloud Visualization**: View the raw 3D points.
5.  **Geometry Mesh**: Convert points into a solid surface (Poisson Reconstruction).
6.  **Textured Mesh (Final)**: Apply high-res textures to the 3D model.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `opencv-python`, `open3d`, `transformers`, `numpy`, `pillow`)*

---

## 🏃 How to Run (Step-by-Step)

### Step 1: Prepare Image
Place your image in the `assets/` folder (e.g., `assets/demo04.jpg`).

### Step 2: Run Depth Estimation (The Core AI)
This script generates the depth map and point cloud.
```bash
python scripts/03_depth_estimation.py --image assets/demo04.jpg
```
*Output: `output/point_cloud_rgb.ply`, `output/depth_map.npy`*

### Step 3: Visualize Point Cloud (Optional)
Inspect the raw 3D points.
```bash
python scripts/04_visualize_point_cloud.py output/point_cloud_rgb.ply
```

### Step 4: Generate Final Textured Mesh
Create the fully textured 3D object (.obj).
```bash
python scripts/06_mesh_textured.py --image assets/demo04.jpg --quality 5
```
*Output: `final_model/model.obj`*

---

## 📂 Output Files

Failed to execute the visualization? Check the `final_model/` folder.
*   **model.obj**: The 3D geometry.
*   **demo04.jpg**: The texture file.
*   **model.mtl**: Material instructions.

**To View:** Drag all three files into [3DViewer.net](https://3dviewer.net/).

---

## 🧠 Technical Details

*   **Model**: Depth Anything V2 (Small)
*   **Technique**: Monocular Depth Estimation -> 2.5D Displacement Mapping -> Textured Mesh.
*   **Limitations**: Generates "Bas-Relief" style 3D (front-facing geometry). Does not hallucinate the back of objects.
