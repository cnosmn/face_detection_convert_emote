# Face Emoji Overlay

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![dlib](https://img.shields.io/badge/dlib-19.x-red.svg)](http://dlib.net/)

## Overview

This project uses computer vision to detect faces in webcam video and overlay them with emoji images in real-time. Using dlib's facial landmark detection, the application identifies face regions and replaces them with randomly selected emoji images, creating an amusing real-time face filter effect.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Functions](#functions)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Features

- Real-time face detection in webcam feed
- Face region replacement with emoji images
- Random emoji selection for variety
- Support for multiple faces simultaneously
- Face landmark detection capability (commented out in current version)
- Image resizing with aspect ratio preservation

## Requirements

- Python 3.x
- OpenCV (cv2)
- dlib
- NumPy
- A webcam
- Facial landmark predictor file (weight.dat)
- Emoji images in a directory named "emotes"

## Installation

1. Install the required Python packages:

```bash
pip install opencv-python numpy dlib argparse
```

2. Download the dlib facial landmark predictor file:
   - You can use the 68-point facial landmark predictor from the dlib website
   - Rename it to "weight.dat" or change the reference in the code

3. Create an "emotes" directory and add your emoji images:
   - Name them numerically (1.jpg, 2.jpg, 3.jpg, etc.)
   - Ensure you have at least 5 emoji images (for the random selection)

## Usage

Run the script to start the webcam feed with emoji overlays:

```bash
python face_emoji_overlay.py
```

Controls:
- Press 'q' or ESC key to exit the application

## How It Works

The application follows these steps:

1. Captures video from the webcam
2. Detects faces using dlib's frontal face detector
3. For each detected face:
   - Converts the bounding box to (x, y, width, height) format
   - Loads a random emoji image
   - Resizes the emoji to match the face dimensions
   - Replaces the face region with the emoji image
4. Displays the modified video feed in real-time

## Functions

### `rect_to_bb(rect)`
Converts dlib's rectangle format to OpenCV's (x, y, width, height) format.

### `shape_to_np(shape, dtype="int")`
Converts dlib's facial landmarks to NumPy array format.

### `resize(image, width=None, height=None, inter=cv2.INTER_AREA)`
Resizes an image with aspect ratio preservation.

### `draw_rectangle(image, gray, rects, predictor, number)`
Processes detected faces and overlays them with emoji images.

### `webcam(predictor_path)`
Main function that handles webcam capture and processes each frame.

## Project Structure

```
project/
│
├── face_emoji_overlay.py     # Main Python script
├── weight.dat                # dlib facial landmark predictor file
└── emotes/                   # Directory containing emoji images
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    └── 5.jpg
```

## Customization

### Using Different Emojis
Replace or add images in the "emotes" directory. The script randomly selects emojis from this directory.

### Camera Selection
Change the webcam index in the `webcam()` function:
```python
cam = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)
```

### Display Facial Landmarks
Uncomment the following code in the `draw_rectangle()` function to display facial landmarks:
```python
for (x,y) in shape:
    cv2.circle(image,(x,y),1,(0,0,255),-1)
```

### Display Face Rectangles
Uncomment the following code in the `draw_rectangle()` function to show face rectangles:
```python
cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.putText(image,"Face #{}".format(i+1),(x-10,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
```

## Troubleshooting

### No Faces Detected
- Ensure proper lighting in your environment
- Check that your webcam is working correctly
- Verify that the "weight.dat" file is correctly loaded

### Emoji Overlay Issues
- Check that your "emotes" directory contains properly named emoji images (1.jpg, 2.jpg, etc.)
- Ensure the emojis are suitable image files (JPG format is recommended)

### Camera Not Found
If you get an error about the camera not being found, try changing the camera index:
```python
cam = cv2.VideoCapture(0)  # Try different indices (0, 1, 2) if needed
```