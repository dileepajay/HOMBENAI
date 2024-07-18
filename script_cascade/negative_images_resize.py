import os
import cv2

# Paths to the directories
current_dir = '../negatives'
resized_dir = os.path.join(current_dir, 'resized')
os.makedirs(resized_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(current_dir) if
               f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.webp')]


# Function to resize image if any side is greater than max_size
def resize_image(image, max_size=64):
    h, w = image.shape[:2]
    if h > max_size or w > max_size:
        # Calculate scaling factor
        scaling_factor = max_size / float(max(h, w))
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    return image


# Resize and rename images
for idx, filename in enumerate(image_files, start=1):
    src_path = os.path.join(current_dir, filename)
    image = cv2.imread(src_path)

    if image is None:
        print(f"Error loading image: {src_path}")
        continue

    # Resize image if needed
    image = resize_image(image, max_size=64)

    # Save the resized image with a new name
    new_name = f'image{idx}.jpg'
    dst_path = os.path.join(resized_dir, new_name)
    cv2.imwrite(dst_path, image)
    print(f"Saved resized image as: {dst_path}")

print("Image resizing and renaming completed.")