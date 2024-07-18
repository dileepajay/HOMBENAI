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

# Directory containing the images
image_dir = '../test_images'

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Loop through the image files
for image_file in image_files:
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

    # Draw rectangles around detected noses
    for (x, y, w, h) in noses:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(f"Drew rectangle at x: {x}, y: {y}, width: {w}, height: {h}")

    # Display the image with detected noses
    print("Displaying the image with detected noses")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
