# 2D to 3D Pipeline: DepthAnything V2

This project provides a complete pipeline to convert 2D images into textured 3D meshes using Monocular Depth Estimation.

## 🚀 Quick Start

### 1. Setup
Make sure you have the dependencies installed:
```bash
pip install -r requirements.txt
```
*(Ensure `opencv-python`, `open3d`, `numpy`, `torch`, `transformers`, `pillow` are installed)*

### 2. Run the Full Pipeline
To convert an image (e.g., `assets/demo04.jpg`) into a 3D model, run these two commands:

**Step 1: Generate Depth Map & Point Cloud**
```bash
python scripts/03_depth_estimation.py --image assets/demo04.jpg
```
*This uses AI to understand the depth of the image.*

**Step 2: Create Textured Mesh**
```bash
python scripts/06_mesh_textured.py --image assets/demo04.jpg --quality 5
```
*This creates the actual 3D object (`.obj`) and packages it for viewing.*

---

## 📂 Output

After running the scripts, check the `final_model/` folder.
You will find three files:
1.  `model.obj` (The 3D Shape)
2.  `model.mtl` (The Material instructions)
3.  `demo04.jpg` (The Texture)

### 3. How to View
Drag **ALL THREE FILES** from the `final_model/` folder onto a 3D viewer website:
*   [3DViewer.net](https://3dviewer.net/)
*   [glTF Viewer](https://gltf-viewer.donmccurdy.com/)

---

## 🛠️ Pipeline Details

1.  **Image Processing (`01-02_*.py`)**
    *   Basic loading and preprocessing of images.
2.  **Depth Estimation (`03_depth_estimation.py`)**
    *   Uses **DepthAnything V2** (Small model) to predict a depth map.
    *   Refines edges using a Grayscale Guided Filter.
    *   Outputs a Point Cloud (`.ply`) and a Raw Depth Map (`.npy`).
3.  **Mesh Generation (`06_mesh_textured.py`)**
    *   Reads the Raw Depth Map.
    *   Creates a Grid Mesh optimized for low-resource systems (CPU friendly).
    *   Maps the original High-Res image as a UV Texture.
    *   Packages the result into a standardized OBJ format.

## ⚠️ Notes
*   This method produces **2.5D Bas-Reliefs**. It does not generate the back of objects (which expects Generative AI).
*   If the model looks too flat or spiky, adjust the `--scale` parameter in `06_mesh_textured.py`.
