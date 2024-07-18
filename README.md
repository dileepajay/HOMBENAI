
# Cow Identity Recognition by Nose-Prints

![Alt text](_docs/app_ui.png) 

This project provides a comprehensive approach to identifying cows by their unique nose-prints. The goal is to develop a reliable system that can recognize individual cows based on the distinctive patterns found on their noses. This method can be highly beneficial for livestock management, ensuring accurate identification without the need for invasive or stressful methods.

### Key Features

- **Image Preprocessing**: The project includes scripts to preprocess images by resizing, grayscale conversion, adaptive thresholding, and morphological operations to highlight nose patterns.
- **Annotation**: Tools to manually annotate images to mark the nose areas, creating a dataset for training.
- **Rescaling**: Automated scripts to rescale images and annotations to multiple sizes, facilitating robust training.
- **Negative Sample Preparation**: Guides to prepare and resize negative samples to improve the model's ability to distinguish noses from other parts of the cow or background.
- **Cascade Classifier Training**: Steps to train a Haar cascade classifier using positive and negative samples, tailored to detect cow noses.
- **Nose Extraction**: Tools to use the trained classifier to extract nose regions from images.
- **Nose-Print Processing**: Scripts to process extracted noses into high-contrast prints suitable for recognition.
- **Identifier Training**: Training an identifier model using KNN or CNN algorithms to recognize individual cows from their nose prints.
- **GUI Application**: A user-friendly graphical interface for testing the system by dragging and dropping test images to see the identification results.

### Current Status

This project is currently in the research and prototype stage. It is not yet intended for industrial use. The current focus is on developing and refining the algorithms and methods to ensure accurate and reliable cow identification. Contributions and feedback are welcome to help improve the system.
 


![Alt text](_docs/cow_node_detection_cascade_test.jpg)

### Installation

1. **Install Python**:
    - Download and install the latest version of Python from [python.org](https://www.python.org/downloads/).
    - Add Python and the Scripts folder to your environment variable `PATH`.

    Example for Windows:
    ```
    C:\Python39;
    C:\Python39\Scripts;
    ```

2. **Install OpenCV**:
    - Download OpenCV  [opencv-3.4.11-vc14_vc15.exe](https://sourceforge.net/projects/opencvlibrary/files/3.4.11/opencv-3.4.11-vc14_vc15.exe/download). This version have pre-builds. 
    - Run the installer and extract the files.
    - Locate the `bin` directory in the extracted files and add it to your environment variable `PATH`.

    Example for Windows:
    ```
    C:\opencv\build\x64\vc15\bin;
    ```

3. **Install Python Dependencies**:
    - Navigate to the project directory and install the required Python packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Project**:
    - You can use the command prompt to run the scripts, or you can use an IDE like PyCharm, VSCode, or any Python-compatible IDE to run the project.

    Example command to run a script:
    ```bash
    python script_name.py
    ```

Replace `script_name.py` with the name of the script you want to run.

## Stages

### Stage 1: Positive Image Samples Preparation

Convert all positive images to a resolution of 1024x1024.

```bash
python download_image_resize.py
```

This script resizes all images in the `positives/` directory and saves them to `positives/1024x1024`.

### Stage 2: Annotate Images

Mark the nose area in all images and create `positives_1024x1024.txt` with image paths and nose coordinates.

```bash
opencv_annotation --annotations=positives_1024x1024.txt --images=positives/1024x1024
```

### Stage 3: Rescale Positive Images

Rescale images and annotations to 512, 256, 128, and 64 sizes.

```bash
python script_cascade/positive_images_resize.py
```

### Stage 4: Prepare Negative Samples

Add samples to the `negatives/` folder. Ensure there are at least twice as many negatives as positives. Then, resize negative images and create `negative.txt`.

```bash
python script_cascade/negative_images_resize.py
python script_cascade/script_create_negative_txt.py
```

### Stage 5: Train Cascade Classifier

Create a vector file and train the cascade classifier.

```bash
opencv_createsamples -info positives_64x64.txt -num 50 -w 24 -h 24 -vec positives_64x64.vec
opencv_traincascade -data script_cascade/output/64 -vec positives_64x64.vec -bg negatives.txt -numPos 50 -numNeg 1600 -numStages 10 -w 24 -h 24 -precalcValBufSize 2048 -precalcIdxBufSize 2048
```

### Stage 6: Test Cascade Classifier

Test the trained cascade classifier.

```bash
python script_cascade/cascade_test.py
python script_cascade/cascade_test_loop.py
```

### Stage 7: Debug and Retrain

If the recognizer misclassifies, add those images to the negatives and retrain from Stage 4.

```bash
python script_cascade/castcade_debug.py
```

### Stage 8: Extract Noses

Use the trained classifier to extract noses from images.

```bash
python script_extract/nose_extract.py
```

### Stage 9: Process Nose-Prints

Grayscale and apply adaptive thresholding to nose-prints.

```bash
python script_extract/node_to_print.py
```

### Stage 10: Train Identifier Model

Train an identifier model using the KNN algorithm (or CNN for better results).

```bash
python script_identifier/train_identifier.py
```

### Stage 11: Test the Model

Test the trained identifier model.

```bash
python script_identifier/model_test.py
```

### Stage 12: Run the App

Run the UI application to test the model. Drag and drop test images to see the labels.

```bash
python app/HOMBENAI.py
```

## Acknowledgements

Thanks to the researchers and authors of the paper "[Cattle identification: the history of nose prints approach in brief](https://www.researchgate.net/publication/347434374_Cattle_identification_the_history_of_nose_prints_approach_in_brief)" for their valuable work and contributions to this field.

## Notes

- Make sure to install all necessary dependencies before running the scripts.
- The model's accuracy can be improved by using a larger dataset and fine-tuning the parameters.

## License

This project is licensed under the MIT License.
