# 2D to 3D Image Conversion Project

This project demonstrates how to convert 2D images into 3D models using the **Depth Anything V2** AI model. It follows a structured learning path from understanding basic image parameters to generating 3D point clouds.

## Project Structure

```
major-project/
├── scripts/
│   ├── 01_image_basics.py       # Module 1: Image Parameters
│   ├── 02_image_processing.py   # Module 2: Image Processing
│   └── 03_depth_estimation.py   # Module 3: Depth Estimation & 3D
├── output/                      # Generated results (depth maps, 3D files)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup

1.  **Create and Activate Virtual Environment**:
    ```bash
    # Create venv
    python -m venv venv
    
    # Activate (Windows)
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Scripts**:
    Now you can run scripts simply with `python`:
    ```bash
    python scripts/03_depth_estimation.py
    ```

## How It Works

### 1. Image Parameters (Module 1)
Understanding the building blocks of an image:
-   **Resolution**: The dimensions (width x height) of the image.
-   **Channels**: Color information (e.g., Red, Green, Blue).
-   **Pixel Values**: The intensity of light at each point (0-255 for 8-bit images).

### 2. Image Processing (Module 2)
Manipulating images before 3D conversion:
-   **Filtering**: Removing noise (e.g., Gaussian Blur) to get cleaner depth maps.
-   **Edge Detection**: Identifying boundaries of objects.

### 3. Depth Estimation (Module 3)
This is the core transformation.
-   **Model**: We use **Depth Anything V2** (via Hugging Face Transformers). It is a Monocular Depth Estimation model, meaning it predicts depth from a single image.
-   **Transformation**: The model takes an RGB image and outputs a **Depth Map**. A depth map is a grayscale image where the pixel value represents the distance from the camera (brighter = closer, darker = further).

### 4. 3D Reconstruction (Module 3)
Converting the 2D Depth Map into 3D space.
-   **Pinhole Camera Model**: We use this mathematical model to project 2D pixels into 3D points $(x, y, z)$.
    -   $z$ comes from the depth map.
    -   $x$ and $y$ are calculated using the pixel coordinates $(u, v)$ and the camera's **Focal Length**.
-   **Hyperparameters**:
    -   **Focal Length**: Determines the field of view. We estimate it as `0.8 * width` in the script.
    -   **Depth Scale**: Scales the relative depth values from the model to real-world units (or consistent relative units).

## Output
-   **Depth Map**: `output/depth_map.png` - Visual representation of depth.
-   **Point Cloud**: `output/point_cloud.ply` - A collection of 3D points representing the scene. You can open this file in MeshLab, CloudCompare, or online 3D viewers.









Preprocessing:
The image is resized to a specific resolution (e.g., 518x518) that the model was trained on.
It is normalized (pixel values scaled to -1 to 1 or 0 to 1).
Forward Pass:
The neural network analyzes the Context and Semantics of the scene.
Example: It recognizes "This is a floor, so it should recede into the distance." "This is a person, so they are likely in the foreground."
It does not use lasers or stereo cameras. It uses learned cues like perspective, occlusion (things blocking other things), and relative size.
Output (Disparity vs. Depth):
The raw output is usually Relative Inverse Depth (Disparity).
High Value = Close to camera.
Low Value = Far from camera.
It is "Relative" because the model doesn't know the real-world scale (meters vs. inches). It only knows "Object A is twice as far as Object B."





The Problem with Standard Grayscale
Usually, when you convert a Color Image to Grayscale (RGB -> Gray), the computer just averages the Red, Green, and Blue values: Gray = (R + G + B) / 3

Issue: This is mathematically simple but perceptually wrong.
Example: A bright red ball on a bright green grass might look exactly the same shade of gray. The edge between the ball and grass disappears because they have the same "intensity," even though they are different colors.
The Solution: LAB Color Space
LAB separates an image into three different channels:

L (Lightness): Pure brightness (Black to White).
A: Green-to-Red color axis.
B: Blue-to-Yellow color axis.
Why we use the L-Channel
The L-Channel is designed to match Human Vision. It knows that Green looks brighter to us than Blue.
By extracting just the L channel, we get a grayscale image that preserves Structural Edges much better than standard grayscale.
In your code:
python
lab_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2LAB) # Convert to LAB
l_channel, _, _ = cv2.split(lab_image)                # Throw away color, keep Lightness
Part 2: Guided Filtering (The "Smart Blur")
The Goal
You have two images:

The Guide (L-Channel): High resolution, sharp edges, perfect detail.
The Input (Depth Map): Low resolution, blurry edges, "cloudy."
You want to transfer the Sharpness of the Guide onto the Input.

How it works (The Math Concept)
The Guided Filter looks at every pixel and asks a question:

"Is this area flat (like a wall) or is it an edge (like a table leg)?"

It looks at the Guide (L-Channel):
If the Guide is Flat (variance is low), the filter says: "Okay, just smooth out the Depth Map here to remove noise."
If the Guide has an Edge (variance is high), the filter says: "STOP! Do not smooth across this line. The Depth Map must change value right here, exactly where the Guide changes."
The Result
The Depth Map inherits the Structure of the original photo.
The blurry blob of a "chair" in the depth map suddenly snaps to the exact pixel outline of the chair in the photo.




python scripts/03_depth_estimation.py --image assets/demo03.jpg --visualize
python scripts/03_depth_estimation.py --visualize