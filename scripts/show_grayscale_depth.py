"""
Simple visualization: Grayscale (L channel) and Depth Map
"""

import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import os

# Load image
image_path = "assets/demo01.jpg"
print(f"Processing: {image_path}")

cv_image = cv2.imread(image_path)
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(cv_image)

# Extract grayscale (L channel from LAB)
print("Extracting L channel (grayscale)...")
lab_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2LAB)
l_channel, _, _ = cv2.split(lab_image)

# Get depth map from AI
print("Generating depth map...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth_result = pipe(pil_image)
depth_map = np.array(depth_result["depth"])

# Resize if needed
if depth_map.shape[:2] != cv_image.shape[:2]:
    depth_map = cv2.resize(depth_map, (cv_image.shape[1], cv_image.shape[0]))

# Normalize depth for visualization
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_norm = (depth_map - depth_min) / (depth_max - depth_min)

# Create visualizations
grayscale_vis = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)
depth_vis = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

# Add labels
cv2.putText(grayscale_vis, "Grayscale (L Channel)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.putText(depth_vis, "Depth Map (AI Output)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

# Stack side-by-side
comparison = np.hstack((grayscale_vis, depth_vis))

# Save
output_path = "output/grayscale_and_depth.png"
os.makedirs("output", exist_ok=True)
cv2.imwrite(output_path, comparison)

print(f"\n✓ Saved to: {output_path}")
print("\nLeft: Grayscale (L channel from LAB color space)")
print("Right: Depth Map (from DepthAnythingV2 AI model)")
