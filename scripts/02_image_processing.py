import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(image_path):
    """
    Demonstrates basic image processing operations: filtering, edge detection, transformations.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Grayscale Conversion
    # Essential for many processing tasks (like edge detection) to reduce complexity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (Filtering)
    # Reduces noise and detail
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Edge Detection (Canny)
    # Detects structural edges in the image
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Simple Transformation (Rotation)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) # Rotate 45 degrees
    rotated = cv2.warpAffine(img_rgb, M, (cols, rows))

    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Canny)")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(rotated)
    plt.title("Rotated 45°")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save the grayscale image for the next step
    output_gray_path = os.path.join("output", "gray_processed.jpg")
    if not os.path.exists("output"):
        os.makedirs("output")
    cv2.imwrite(output_gray_path, gray)
    print(f"Saved grayscale image to {output_gray_path}")
    return output_gray_path

if __name__ == "__main__":
    # Use a demo image from assets
    image_path = os.path.join("assets", "demo03.jpg")
    
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Creating dummy.")
        image_path = "sample_image.jpg"
        if not os.path.exists(image_path):
            import os
            gradient = np.zeros((400, 400, 3), dtype=np.uint8)
            for i in range(400):
                gradient[i, :, :] = i % 255
            cv2.imwrite(image_path, gradient)
    
    process_image(image_path)
