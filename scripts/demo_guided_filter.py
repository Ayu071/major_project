"""
Simple demonstration of Guided Filter effect on depth maps.
This creates a visual comparison showing how the grayscale guide image
improves depth map quality.
"""

import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import os

def guided_filter(I, p, r, eps):
    """Guided Filter implementation"""
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    
    q = mean_a * I + mean_b
    return q

# Load image
image_path = "assets/demo01.jpg"
print(f"Processing {image_path}...")

cv_image = cv2.imread(image_path)
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(cv_image)

# Extract grayscale guide (LAB L channel)
lab_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2LAB)
l_channel, _, _ = cv2.split(lab_image)
guide_image = l_channel.astype(np.float32) / 255.0

# Get depth map from AI
print("Running AI depth estimation...")
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth_result = pipe(pil_image)
depth_map = np.array(depth_result["depth"])

# Resize if needed
if depth_map.shape[:2] != cv_image.shape[:2]:
    depth_map = cv2.resize(depth_map, (cv_image.shape[1], cv_image.shape[0]))

# Normalize
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_norm = (depth_map - depth_min) / (depth_max - depth_min)

# Create "noisy" version with MUCH MORE aggressive degradation
# This will make the difference very obvious
print("Creating comparison with aggressive degradation...")

# Step 1: Heavy downsampling (lose all edge detail)
h, w = depth_norm.shape
depth_degraded = cv2.resize(depth_norm, (w//8, h//8))  # Downsample to 1/8 size
depth_degraded = cv2.resize(depth_degraded, (w, h))    # Upscale back (creates blocky artifacts)

# Step 2: Add significant noise
noise = np.random.normal(0, 0.08, depth_degraded.shape).astype(np.float32)
depth_degraded = np.clip(depth_degraded + noise, 0, 1)

# Step 3: Heavy blur to make edges very soft
depth_degraded = cv2.GaussianBlur(depth_degraded, (21, 21), 0)

print("Applying guided filter to repair the degraded depth...")
# Apply guided filter with aggressive parameters
depth_refined = guided_filter(guide_image, depth_degraded, 16, 0.001)

# Create visualization
d_degraded_vis = cv2.applyColorMap((depth_degraded * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
d_refined_vis = cv2.applyColorMap((depth_refined * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
guide_vis = cv2.cvtColor((guide_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Add labels
cv2.putText(guide_vis, "Grayscale Guide (L channel)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(d_degraded_vis, "Before: Heavily Degraded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(d_refined_vis, "After: Guided Filter Repair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Create comparison grid
row1 = np.hstack((cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR), guide_vis))
row2 = np.hstack((d_degraded_vis, d_refined_vis))
comparison = np.vstack((row1, row2))

# Save
output_path = "output/guided_filter_demo.png"
os.makedirs("output", exist_ok=True)
cv2.imwrite(output_path, comparison)

print(f"\n✓ Saved comparison to: {output_path}")
print("\nWhat you're seeing:")
print("  Top-left: Original image")
print("  Top-right: Grayscale guide (L channel from LAB color space)")
print("  Bottom-left: HEAVILY degraded depth (downsampled 8x, noisy, blurred)")
print("  Bottom-right: Repaired using guided filter with grayscale guide")
print("\nThe difference should be VERY obvious now!")
print("The guided filter uses the sharp edges from the grayscale guide")
print("to recover detail from the heavily degraded depth map.")
