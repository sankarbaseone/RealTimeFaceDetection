# Real-Time Face Detection with Head Pose Estimation

A real-time face detection and head pose estimation application using MediaPipe and OpenCV. This application detects faces from your webcam feed and calculates head orientation angles (pitch, yaw, roll) in real-time.

## Features

- ✅ Real-time face detection using MediaPipe Face Mesh
- ✅ Head pose estimation with 3D orientation angles:
  - **Pitch** (up/down rotation)
  - **Yaw** (left/right rotation)
  - **Roll** (tilt rotation)
- ✅ 468 facial landmarks visualization
- ✅ 3D axes overlay on face (X, Y, Z axes)
- ✅ Head orientation text (e.g., "Looking Left & Up")
- ✅ FPS counter
- ✅ Mirror view for natural interaction

## Demo

The application displays:
- Face mesh overlay with 468 3D landmarks
- 3D axes (red, green, blue) drawn from the nose tip
- Real-time pitch, yaw, and roll angles in degrees
- Textual head orientation description
- FPS and face count

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/RealTimeFaceDetection.git
cd RealTimeFaceDetection
```

2. Install required packages:
```bash
pip install mediapipe opencv-python-headless==4.7.0.72 numpy
```

## Usage

Run the application:
```bash
python cam.py
```

**Controls:**
- Press **'q'** to quit the application

## How It Works

1. **Face Detection**: Uses MediaPipe Face Mesh to detect 468 3D facial landmarks
2. **Head Pose Estimation**: Applies the PnP (Perspective-n-Point) algorithm to estimate head rotation
3. **Euler Angles**: Calculates pitch, yaw, and roll from the rotation matrix
4. **Visualization**: Draws 3D axes and displays orientation information in real-time

### Head Pose Angles Explained

- **Pitch**: Rotation around the X-axis (nodding up/down)
  - Positive: Looking down
  - Negative: Looking up
  
- **Yaw**: Rotation around the Y-axis (turning left/right)
  - Positive: Looking left
  - Negative: Looking right
  
- **Roll**: Rotation around the Z-axis (tilting head)
  - Positive: Tilting head to the right
  - Negative: Tilting head to the left

## Technical Details

- **Face Mesh Model**: MediaPipe Face Mesh with 468 landmarks
- **Camera Matrix**: Approximated using frame width as focal length
- **PnP Algorithm**: OpenCV's `solvePnP` with iterative method
- **3D Model Points**: 6 key facial landmarks (nose tip, chin, eye corners, mouth corners)

## Project Structure

```
RealTimeFaceDetection/
├── cam.py          # Main application script
├── README.md       # This file
└── .gitignore      # Git ignore file
```

## License

MIT License - feel free to use this project for learning and development.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) by Google for face mesh detection
- [OpenCV](https://opencv.org/) for computer vision utilities
