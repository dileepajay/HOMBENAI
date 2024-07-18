import os
import cv2

# Base directory
base_dir = '../positives'

# Target sizes and directories
sizes = [512, 256, 128, 64, 32]
target_dirs = [os.path.join(base_dir, f"{size}x{size}") for size in sizes]

# Create target directories
for dir in target_dirs:
    os.makedirs(dir, exist_ok=True)


# Function to resize image
def resize_image(image, size):
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# Load the annotations
annotations_file = '../positives_1024x1024.txt'
with open(annotations_file, 'r') as file:
    annotations = file.readlines()

# Process images and annotations for each size
for size, target_dir in zip(sizes, target_dirs):
    new_annotations = []
    for line in annotations:
        parts = line.strip().split()
        image_path = parts[0]
        num_objects = parts[1]
        x, y, w, h = map(int, parts[2:])

        # Load image
        image_name = os.path.basename(image_path)
        src_path = os.path.join(base_dir, '1024x1024', image_name)
        image = cv2.imread(src_path)

        if image is None:
            print(f"Error loading image: {src_path}")
            continue

        # Resize image
        resized_image = resize_image(image, size)

        # Calculate new bounding box
        x = int(x * size / 1024)
        y = int(y * size / 1024)
        w = int(w * size / 1024)
        h = int(h * size / 1024)

        # Save resized image
        dst_path = os.path.join(target_dir, image_name)
        cv2.imwrite(dst_path, resized_image)

        # Create new annotation
        new_annotations.append(f"{image_path} {num_objects} {x} {y} {w} {h}\n")

    # Save new annotations to file
    annotations_file = f'../positives_{size}x{size}.txt'
    with open(annotations_file, 'w') as file:
        file.writelines(new_annotations)

    print(f"Processed images and annotations for size {size}x{size}")

print("Image processing and annotation creation completed.")