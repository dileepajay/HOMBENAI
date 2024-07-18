import os
import cv2
from matplotlib import pyplot as plt

# Load the trained cascade classifier
cascade_path = 'output/64/cascade.xml'
print(f"Loading cascade classifier from: {cascade_path}")
nose_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade was loaded correctly
if nose_cascade.empty():
    print("Error loading cascade classifier")
else:
    print("Cascade classifier loaded successfully")

# Directories
image_dir = '../test_images'
negatives_dir = '../negatives'
os.makedirs(negatives_dir, exist_ok=True)

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# List to store the paths of saved negative images
negative_image_paths = []

# Loop through the image files
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    print(f"\nLoading image from: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    else:
        print("Image loaded successfully")

    # Convert the image to grayscale
    print("Converting image to grayscale")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect noses
    print("Detecting noses in the image")
    noses = nose_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any noses are detected
    if len(noses) == 0:
        print("No noses detected")
    else:
        print(f"Detected {len(noses)} nose(s)")

        # Save each detected nose as a separate image
        for (x, y, w, h) in noses:
            nose_image = image[y:y+h, x:x+w]
            negative_image_path = os.path.join(negatives_dir, f'image{idx + 1}_{x}_{y}.jpg')
            cv2.imwrite(negative_image_path, nose_image)
            negative_image_paths.append(os.path.join('negatives\\resized\\', f'image{idx + 1}_{x}_{y}.jpg'))
            print(f"Saved detected nose as: {negative_image_path}")

# Write the list of negative images to negative.txt
negative_txt_path = 'negative.txt'
with open(negative_txt_path, 'w') as f:
    for path in negative_image_paths:
        f.write(f"{path}\n")

print(f"Negative images saved and listed in: {negative_txt_path}")
