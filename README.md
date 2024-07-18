
# Cow Identity Recognition by Nose-Prints

This project provides a comprehensive approach to identifying cows by their unique nose-prints. It involves multiple stages including image preprocessing, feature extraction, classifier training, and testing.

## Project Directory Structure

```
cow_identity_recognition/
├── app/
│   └── HOMBENAI.py
├── script_cascade/
│   ├── cascade_test.py
│   ├── cascade_test_loop.py
│   ├── castcade_debug.py
│   ├── negative_images_resize.py
│   ├── positive_images_resize.py
│   ├── script_create_negative_txt.py
│   └── output/
│       └── 64/
├── script_extract/
│   ├── nose_extract.py
│   └── node_to_print.py
├── script_identifier/
│   ├── train_identifier.py
│   ├── model_test.py
│   └── output/
├── positives/
│   └── 1024x1024/
├── negatives/
├── noses/
├── noseprint/
├── test_images/
├── README.md
```

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
