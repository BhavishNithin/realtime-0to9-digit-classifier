# Live Camera Digit Recognition (0-9)

This repository contains code for real-time recognition of single digits (0-9) from a live camera feed. It leverages OpenCV for video capture and image processing, and a TensorFlow model for digit classification.

## Overview

The application captures video from your webcam, isolates a defined region of interest (ROI), and uses a pre-trained TensorFlow neural network to predict the digit within that ROI. The predicted digit is displayed on the video feed.

## Features

* **Real-time digit recognition:** Processes video frames to predict digits as they appear in the ROI.
* **Digit classification (0-9):** Specifically trained to identify single digits from zero to nine.
* **Webcam integration:** Uses OpenCV to capture live video from your default camera.
* **Region of Interest (ROI):** Focuses prediction on a specific area in the camera frame.
* **Threshold adjustment:** Includes a trackbar for dynamically adjusting the binary threshold used for image segmentation.
* **Visual feedback:** Displays the predicted digit and the ROI on the video feed.
* **Edge visualization:** Shows the thresholded image in a separate window for better understanding of the segmentation.

## Prerequisites

* Python 3.x
* TensorFlow (`tensorflow`)
* NumPy (`numpy`)
* OpenCV (`opencv-python`)

You can install these dependencies using pip:

```bash
pip install tensorflow numpy opencv-python
