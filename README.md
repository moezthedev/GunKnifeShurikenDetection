# Object Detection Web App

This is a simple web application built using Flask that performs object detection on uploaded images. The application uses the SIFT (Scale-Invariant Feature Transform) algorithm to detect and match features of objects in images.

## Features

- **Image Upload**: Users can upload images through a web interface.
- **Object Detection**: Detects and identifies predefined objects (gun, knife, shuriken) using SIFT feature descriptors.
- **Results Visualization**: Highlights detected objects in the uploaded image and displays the result.

## Technologies Used

- **Flask**: Web framework for Python.
- **OpenCV**: Library for computer vision tasks.
- **NumPy**: Library for numerical operations in Python.

## How It Works

1. **Upload Image**: Users upload an image file (PNG, JPG, or JPEG) through the web interface.
2. **Object Detection**: The application processes the uploaded image to detect objects. It uses SIFT to extract and match feature descriptors against predefined templates (gun, knife, shuriken).
3. **Display Results**: Detected objects are highlighted with different colors and the result is displayed.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
