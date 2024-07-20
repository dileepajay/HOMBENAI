import cv2
import numpy as np

# Load the trained model and label dictionary
model_path = r"output\\cowrec_knn_model.xml"
knn = cv2.ml.KNearest_load(model_path)
label_dict_path = r"output\\label_dict.npy"
label_dict = np.load(label_dict_path, allow_pickle=True).item()
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Initialize ORB detector
orb = cv2.ORB_create()

# Path to test image
test_img_path = r"../noseprint/image2.jpg"

# Read and preprocess the test image
test_img = cv2.imread(test_img_path)
test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Detect and compute features
keypoints, test_des = orb.detectAndCompute(test_img_gray, None)

# Predict using the trained model
if test_des is not None:
    ret, results, neighbours, dist = knn.findNearest(test_des.astype(np.float32), k=3)
    label_counts = np.bincount(results.flatten().astype(int))
    predicted_label_id = np.argmax(label_counts)
    predicted_label = reverse_label_dict[predicted_label_id]
    print(f"The detected image ID is: {predicted_label}")
else:
    print("No features detected in the test image.")
