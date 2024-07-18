import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Paths to the directories
input_dir = '../noses'
output_dir = '../noseprint'
os.makedirs(output_dir, exist_ok=True)

# Function to process and convert the image
def convert_image(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the binary image
    inverted_image = cv2.bitwise_not(adaptive_thresh)

    # Apply morphological operations to enhance the patterns
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

    # Save the processed image
    cv2.imwrite(output_path, cv2.resize(processed_image, (256, 256)))

    print(f"Processed and saved: {output_path}")

# Get list of image files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Process each image
for image_file in image_files:
    input_path = os.path.join(input_dir, image_file)
    output_path = os.path.join(output_dir, image_file)
    convert_image(input_path, output_path)

print("Image processing completed.")
