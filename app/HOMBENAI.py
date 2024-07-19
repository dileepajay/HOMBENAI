import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import time

# Load the trained model and label dictionary
model_path = r"..//script_identifier//output//cowrec_knn_model.xml"
knn = cv2.ml.KNearest_load(model_path)
label_dict_path = r"..//script_identifier//output//label_dict.npy"
label_dict = np.load(label_dict_path, allow_pickle=True).item()
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Initialize ORB detector
orb = cv2.ORB_create()


# Function to process and convert the image
def convert_image(image):
    # Read the image

    image = cv2.imread('temp_.jpg')

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

    processed_image = cv2.resize(processed_image, (256, 256))

    print(f"Processed and saved: temp.jpg")

    return processed_image


# Function to handle file drop
def on_drop(event):
    file_path = event.data
    file_path = file_path.strip('{}')  # Remove curly braces if present
    cascade_path = '..//script_cascade//output//64//cascade.xml'

    # Process the dropped image
    original_image, cropped_nose, processed_image = process_image(file_path, cascade_path)

    # Display images
    display_image(original_image, original_label)
    display_image(cropped_nose, cropped_label)
    display_image(processed_image, processed_label)

    # Predict the class of the processed image using ORB and KNN
    # test_img = cv2.imread('temp.jpg')
    # test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('temp.jpg', processed_image)
    keypoints, test_des = orb.detectAndCompute(processed_image, None)

    if test_des is not None:
        ret, results, neighbours, dist = knn.findNearest(test_des.astype(np.float32), k=3)
        label_counts = np.bincount(results.flatten().astype(int))
        print(f"label_counts {label_counts}")
        predicted_label_id = np.argmax(label_counts)
        print(f"predicted_label_id {predicted_label_id}")
        print(f"reverse_label_dict {reverse_label_dict}")

        predicted_label = reverse_label_dict[predicted_label_id]
        result_text = f"Detected Cow: {predicted_label}"
        result_label.config(text=result_text, bg='green', font=("Helvetica", 20))
    else:
        result_label.config(text="Not recognized", bg='red', font=("Helvetica", 20))


# Function to process the image
def process_image(image_path, cascade_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the trained cascade classifier
    nose_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect noses
    noses = nose_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the largest detection
    def get_largest_detection(detections):
        if len(detections) == 0:
            return None
        return max(detections, key=lambda rect: rect[2] * rect[3])

    largest_nose = get_largest_detection(noses)

    if largest_nose is None:
        cropped_nose = np.zeros_like(image)
        processed_image = np.zeros_like(image)
    else:
        x, y, w, h = largest_nose
        cropped_nose = image[y:y + h, x:x + w]

        # Save the processed image
        cv2.imwrite('temp_.jpg', cropped_nose)

        # Process the cropped nose image
        processed_image = convert_image(cropped_nose)

    return image, cropped_nose, processed_image


# Function to display an image on a label
def display_image(cv_img, label):
    # Convert the image to RGB format
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_img = Image.fromarray(cv_img_rgb)
    # Resize the image to fit the label
    pil_img = pil_img.resize((250, 250), Image.LANCZOS)
    # Convert to ImageTk
    imgtk = ImageTk.PhotoImage(image=pil_img)
    label.config(image=imgtk)
    label.image = imgtk


# Set up the GUI
root = TkinterDnD.Tk()
root.title("HOMBENAI by dileepajay")
root.geometry("800x600")

# Labels to display images
original_label = tk.Label(root)
original_label.grid(row=0, column=0, padx=10, pady=10)
cropped_label = tk.Label(root)
cropped_label.grid(row=0, column=1, padx=10, pady=10)
processed_label = tk.Label(root)
processed_label.grid(row=0, column=2, padx=10, pady=10)

# Label to display the result
result_label = tk.Label(root, text="Result will be shown here", font=("Helvetica", 16), width=40, height=2)
result_label.grid(row=2, column=0, columnspan=3, pady=10)

# Instructions
instructions = tk.Label(root, text="Drag and drop an image file into the window", font=("Helvetica", 16))
instructions.grid(row=1, column=0, columnspan=3, pady=10)

# Enable drag and drop
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

# Start the GUI event loop
root.mainloop()
