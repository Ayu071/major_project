import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 Major Project: 3D Generation with TripoSR (Colab GPU Version)\n",
                    "\n",
                    "This notebook uses the transformer-based **TripoSR** model to generate full 3D from a single image.\n",
                    "**Crucial:** Ensure Runtime -> Change Runtime Type -> T4 GPU is selected."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Install Dependencies\n",
                    "# We must install EVERYTHING TripoSR needs, including heavy libraries.\n",
                    "!pip install --upgrade pip\n",
                    "# Install PyTorch with CUDA explicitly just in case colab defaults are weird\n",
                    "!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n",
                    "# Install the main TripoSR repo\n",
                    "!pip install git+https://github.com/VAST-AI-Research/TripoSR.git\n",
                    "# Install critical helpers (RemBG for background, XAtlas for UVs, Trimesh for mesh)\n",
                    "!pip install imageio[ffmpeg] rembg[gpu] trimesh xatlas omegaconf  einops"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 2. Clone Repository & Setup\n",
                    "import os\n",
                    "if not os.path.exists('TripoSR'):\n",
                    "    !git clone https://github.com/VAST-AI-Research/TripoSR.git\n",
                    "    %cd TripoSR\n",
                    "    # We create a dummy folder so the script doesn't crash looking for old checkpoints\n",
                    "    !mkdir -p checkpoints\n",
                    "    \n",
                    "print(\"Setup Complete!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Upload Image"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "import shutil\n",
                    "\n",
                    "uploaded = files.upload()\n",
                    "filename = list(uploaded.keys())[0]\n",
                    "print(f\"Processing: {filename}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Run Generation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run TripoSR inference\n",
                    "# This script handles background removal and 3D generation\n",
                    "!python run.py \"../{filename}\" --output-dir \"../output/\""
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Download 3D Model"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "from google.colab import files\n",
                    "\n",
                    "output_dir = '../output'\n",
                    "if os.path.exists(output_dir):\n",
                    "    # Zip the folder to serve as a download\n",
                    "    !zip -r result.zip ../output\n",
                    "    files.download('result.zip')\n",
                    "else:\n",
                    "    print(\"Output folder not found.\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("Project_Notebook_Tripo.ipynb", "w") as f:
        json.dump(notebook, f, indent=4)
    
    print("✓ Created: Project_Notebook_Tripo.ipynb (Fixed xatlas dependency)")

if __name__ == "__main__":
    create_notebook()
