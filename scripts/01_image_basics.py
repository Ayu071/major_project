import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def explore_image_parameters(image_path):
    """
    Explores and prints basic parameters of an image.
    """
    # Load image using OpenCV
    # OpenCV loads images in BGR format by default
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert BGR to RGB for correct visualization with Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"--- Image Parameters for: {os.path.basename(image_path)} ---")
    
    # 1. Resolution (Dimensions)
    height, width, channels = img.shape
    print(f"Resolution: {width}x{height} pixels")
    print(f"Height: {height}, Width: {width}")
    
    # 2. Color Channels
    print(f"Channels: {channels} (e.g., 3 for RGB)")
    
    # 3. Pixel Values (Data Type and Range)
    print(f"Data Type: {img.dtype}")
    print(f"Min Pixel Value: {img.min()}")
    print(f"Max Pixel Value: {img.max()}")
    
    # 4. Brightness and Contrast (Simple stats)
    mean_brightness = np.mean(img)
    std_contrast = np.std(img)
    print(f"Average Brightness: {mean_brightness:.2f}")
    print(f"Contrast (Std Dev): {std_contrast:.2f}")

    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Histogram of pixel intensities
    plt.subplot(1, 2, 2)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title("Color Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Use a demo image from assets
    image_path = os.path.join("assets", "demo01.jpg")
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found. Creating dummy.")
        dummy_path = "sample_image.jpg"
        if not os.path.exists(dummy_path):
            # Create a simple gradient image
            gradient = np.zeros((400, 400, 3), dtype=np.uint8)
            for i in range(400):
                gradient[i, :, :] = i % 255
            cv2.imwrite(dummy_path, gradient)
        image_path = dummy_path

    explore_image_parameters(image_path)
