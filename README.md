# Real-Time Face Detection & Head Pose Estimation

A real-time face detection, head pose estimation, and gaze tracking application. This project now includes a Flask-based Web UI and supports multiple detection methods (MediaPipe, YOLO, Haar Cascade).

## Features

- **Face Detection**: Real-time face detection using Haar Cascade or MediaPipe.
- **Head Pose Estimation**: Calculates Pitch, Yaw, and Roll angles with 3D axis visualization.
- **Gaze Tracking**: Detects eye movement direction (Left, Right, Up, Down, Center).
- **Web UI**: Modern, responsive web interface streaming the live video feed.
- **Motion Tracking**: Visualizes face movement with motion trails.
- **Pupil Detection**: Tracks pupil position within the eye.

## Tech Stack

- **Python 3.7+**
- **OpenCV**: Computer vision and image processing.
- **Flask**: Web server for the UI.
- **MediaPipe**: (Optional) For mesh-based face detection.
- **YOLOv8**: (Optional) For object/face detection.
- **NumPy**: Numerical operations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sankarbaseone/RealTimeFaceDetection.git
   cd RealTimeFaceDetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web UI (Recommended)
Run the Flask application to view the detection in your browser with a modern UI.
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:5000`

### Standalone Scripts
- **MediaPipe Version**:
  ```bash
  python cam.py
  ```
- **YOLO/Haar Cascade Version**:
  ```bash
  python cam_yolo.py
  ```

## Controls
- Press **'q'** to quit the standalone scripts.
- For the Web UI, simply close the browser tab or stop the server (`Ctrl+C`).

## How It Works

1. **Face Detection**: Identifies faces in the video stream.
2. **Landmark/Eye Detection**: Locates eyes and key facial features.
3. **Pose Calculation**: Uses PnP algorithm or bounding box geometry to estimate head orientation.
4. **Gaze Analysis**: Thresholding and contour analysis to find pupils and determine gaze direction.


