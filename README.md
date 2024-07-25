# Face Mask Detection

## Overview

This project is designed to detect whether individuals are wearing face masks or not using a machine learning model. The model is built with Convolutional Neural Networks (CNN) and MobileNetV2 architecture, achieving an impressive accuracy of 99%.

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

## Project Description

The Face Mask Detection system uses a convolutional neural network (CNN) with MobileNetV2 as the base model to classify images into two categories: with mask and without mask. The project includes a real-time detection application using a webcam to identify mask-wearing status.

## Features

- **High Accuracy:** Achieved 99% accuracy in mask detection.
- **Real-time Detection:** Capable of detecting masks in real-time through a video feed.
- **Data Augmentation:** Utilizes image data augmentation to enhance model performance.

## Technology Stack

- **Programming Language:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, imutils, NumPy, Pandas, Matplotlib
- **Machine Learning Framework:** Convolutional Neural Networks (CNN)
- **Base Model:** MobileNetV2

## Dataset

The dataset consists of images categorized into two classes:
- `with_mask`
- `without_mask`

The dataset is used to train and test the model's ability to distinguish between masked and unmasked faces.

- **Dataset Path:** `C:\Users\swapn\Machine learning projects\PREPINSTA\DATASET\FACE MASK VEDIO DETECTION PROJECT`
- **Number of Images:** [Insert Number of Images]
- **Image Dimensions:** 224x224

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Swapnilghait/Face-Mask-Detection.git
    cd Face-Mask-Detection
    ```

2. **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, install the required libraries manually:
    ```bash
    pip install imutils opencv-python tensorflow keras
    ```

## Usage

To use the face mask detection model, follow these instructions:

1. **Prepare the dataset:** Ensure that your dataset is organized into `with_mask` and `without_mask` directories.

2. **Train the model:**
    ```bash
    python train_model.py
    ```

3. **Run real-time detection:**
    ```bash
    python detect_mask.py
    ```

    This script will start a video stream from your webcam and display the mask detection results in real-time.

## Results

The trained model achieved 99% accuracy on the test dataset. The real-time detection script can identify whether individuals are wearing a mask with high precision.

![Sample Results](path_to_sample_image)

## Future Work

- **Enhance Accuracy:** Experiment with different architectures and hyperparameters to improve accuracy.
- **Real-world Deployment:** Adapt the model for use in public spaces and integrate it into larger systems.
- **Expand Dataset:** Include more diverse images to increase the robustness of the model.

## Contributing

Contributions are welcome! If you have any ideas or improvements, please fork the repository and submit a pull request. For significant changes, please open an issue to discuss your proposal.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out:

- **Email:** swapnilghait@gmail.com
- **GitHub:** [Swapnilghait](https://github.com/Swapnilghait)

Thank you for checking out my Face Mask Detection project!

