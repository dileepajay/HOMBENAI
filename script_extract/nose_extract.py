import os
import cv2
from matplotlib import pyplot as plt

# Load the trained cascade classifier
cascade_path = '../script_cascade/output/64/cascade.xml'
print(f"Loading cascade classifier from: {cascade_path}")
nose_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade was loaded correctly
if nose_cascade.empty():
    print("Error loading cascade classifier")
else:
    print("Cascade classifier loaded successfully")

# Directory containing the images
image_dir = '../test_images'
output_dir = '../noses'
os.makedirs(output_dir, exist_ok=True)

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

def get_largest_detection(detections):
    if len(detections) == 0:
        return None
    return max(detections, key=lambda rect: rect[2] * rect[3])

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

    # Get the largest detection
    largest_nose = get_largest_detection(noses)

    if largest_nose is None:
        print("No noses detected")
    else:
        x, y, w, h = largest_nose
        cropped_nose = image[y:y+h, x:x+w]
        print(f"Cropped nose at x: {x}, y: {y}, width: {w}, height: {h}")

        # Save the cropped nose image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, cropped_nose)
        print(f"Saved cropped nose image to: {output_path}")

        # Display the cropped nose image
        print("Displaying the cropped nose image")
        plt.imshow(cv2.cvtColor(cropped_nose, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
