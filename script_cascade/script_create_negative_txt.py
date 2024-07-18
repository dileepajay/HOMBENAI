import os

# Define the paths
negatives_dir = '../negatives/resized'
negatives_txt_path = '../negatives.txt'

# Get list of image files in the negatives directory
image_files = [f for f in os.listdir(negatives_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Create the negatives.txt file
with open(negatives_txt_path, 'w') as f:
    for image_file in image_files:
        image_path = os.path.join('negatives\\resized', image_file)
        f.write(f"{image_path}\n")
        print(f"Added {image_path} to negatives.txt")

print(f"negatives.txt file created at {negatives_txt_path}")