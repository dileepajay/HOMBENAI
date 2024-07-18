import os
import cv2
import numpy as np

# Paths to the directories
current_dir = '../positives'
output_dir = os.path.join(current_dir, '1024x1024')
os.makedirs(output_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(current_dir) if
               f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.webp')]


# Function to crop the maximum 1:1 region from the center
def crop_center(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image[top:top + min_dim, left:left + min_dim]


# Function to resize image to 1024x1024
def resize_image(image, size=(1024, 1024)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# Process images
for idx, filename in enumerate(image_files, start=1):
    src_path = os.path.join(current_dir, filename)
    image = cv2.imread(src_path)

    if image is None:
        print(f"Error loading image: {src_path}")
        continue

    # Crop and resize the image
    cropped_image = crop_center(image)
    resized_image = resize_image(cropped_image, size=(1024, 1024))

    # Save the processed image with a new name
    new_name = f'image{idx}.jpg'
    dst_path = os.path.join(output_dir, new_name)
    cv2.imwrite(dst_path, resized_image)
    print(f"Processed and saved image as: {dst_path}")

print("Image processing completed.")