import cv2
import numpy as np
import os
import re

# Directory containing training images
train_dir = r"../noseprint"
model_dir = r"output"

# Ensure the model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize ORB detector
orb = cv2.ORB_create()

# Initialize lists for descriptors and labels
descriptors = []
labels = []
label_dict = {}
label_id = 0

# Labels
label_names = ["නුවන්ගේ", "සුමනෙගෙ", "දිනපාලගෙ", "විමලගෙ", "සෝමෙගෙ", "අමරෙගෙ",
               "මහින්දගෙ", "රනිල්ගෙ", "සජිත්ගෙ", "අනුරගෙ", "තිලකෙගෙ", "බන්ඩගෙ",
               "දිල්ශාන්ගෙ", "කාලිගෙ"]

n = 0

# Iterate through training images
for filename in os.listdir(train_dir):
    if filename.endswith(".jpg"):
        # Read image
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)

        # Detect and compute features
        keypoints, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
            if n not in label_dict:
                label_dict[label_names[n]] = n
            labels.extend([n] * len(des))
            n = n + 1

# Convert to numpy arrays
descriptors = np.vstack(descriptors)
labels = np.array(labels)

# Train k-NN classifier
knn = cv2.ml.KNearest_create()
knn.train(descriptors.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

# Save the trained model and label dictionary
model_path = os.path.join(model_dir, "cowrec_knn_model.xml")
knn.save(model_path)
label_dict_path = os.path.join(model_dir, "label_dict.npy")
np.save(label_dict_path, label_dict)

print("Training completed and model saved.")
