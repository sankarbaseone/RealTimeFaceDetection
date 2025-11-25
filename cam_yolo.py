import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model for person detection
# We'll use this to detect people and focus on head region
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')  # YOLOv8 nano model (auto-downloads)

# Also load Haar Cascade for face detection (backup/refinement)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS calculation variables
prev_time = 0
curr_time = 0

# Tracking parameters
track_history = {}  # Store tracking history for each face
face_id_counter = 0
tracked_faces = {}

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Distortion coefficients
dist_coeffs = np.zeros((4, 1))

def get_camera_matrix(frame_width, frame_height):
    """Get camera matrix for head pose estimation"""
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    return camera_matrix

def calculate_head_pose_from_bbox(bbox, frame_width, frame_height):
    """
    Estimate head pose from bounding box
    This is a simplified approach using bbox geometry
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate face center and dimensions
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Estimate yaw based on face position in frame
    frame_center_x = frame_width / 2
    yaw = ((face_center_x - frame_center_x) / frame_center_x) * 45  # Max 45 degrees
    
    # Estimate pitch based on vertical position
    frame_center_y = frame_height / 2
    pitch = ((face_center_y - frame_center_y) / frame_center_y) * 30  # Max 30 degrees
    
    # Estimate roll based on aspect ratio (simplified)
    aspect_ratio = face_width / face_height if face_height > 0 else 1
    roll = (aspect_ratio - 1) * 20  # Simplified roll estimation
    
    return pitch, yaw, roll

def draw_axis(frame, yaw, pitch, roll, tdx, tdy, size=100):
    """Draw 3D axes on the face to visualize head pose"""
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    
    # X-axis (red)
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
    cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    
    # Y-axis (green)
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
    cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    # Z-axis (blue)
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)

def get_orientation_text(yaw, pitch):
    """Get human-readable orientation text"""
    orientation = ""
    if abs(yaw) < 10:
        orientation = "Looking Forward"
    elif yaw > 10:
        orientation = "Looking Left"
    else:
        orientation = "Looking Right"
    
    if pitch > 15:
        orientation += " & Down"
    elif pitch < -15:
        orientation += " & Up"
    
    return orientation

print("Starting YOLO + Haar Cascade face detection with tracking... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera")
        break
    
    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    
    frame_height, frame_width, _ = frame.shape
    
    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade (fast and reliable for faces)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process detected faces
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Calculate face center
            face_center_x = int((x1 + x2) / 2)
            face_center_y = int((y1 + y2) / 2)
            
            # Draw bounding box with rounded corners effect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw face ID
            face_id = i + 1
            cv2.putText(frame, f'Face {face_id}', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate head pose
            pitch, yaw, roll = calculate_head_pose_from_bbox(
                [x1, y1, x2, y2], frame_width, frame_height
            )
            
            # Draw 3D axes
            draw_axis(frame, yaw, pitch, roll, face_center_x, face_center_y, size=80)
            
            # Display head pose angles (for the first face only, to avoid clutter)
            if i == 0:
                y_offset = 30
                cv2.putText(frame, f'Pitch: {int(pitch)}°', (frame_width - 250, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'Yaw: {int(yaw)}°', (frame_width - 250, y_offset + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f'Roll: {int(roll)}°', (frame_width - 250, y_offset + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display orientation text
                orientation = get_orientation_text(yaw, pitch)
                cv2.putText(frame, orientation, (frame_width - 250, y_offset + 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Store tracking history
            if face_id not in track_history:
                track_history[face_id] = []
            track_history[face_id].append((face_center_x, face_center_y))
            
            # Keep only last 30 points
            if len(track_history[face_id]) > 30:
                track_history[face_id].pop(0)
            
            # Draw tracking trail
            if face_id in track_history:
                points = track_history[face_id]
                for j in range(1, len(points)):
                    thickness = int(np.sqrt(30 / float(j + 1)) * 2)
                    cv2.line(frame, points[j - 1], points[j], (255, 0, 255), thickness)
    
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display number of faces detected
    num_faces = len(faces)
    cv2.putText(frame, f'Faces: {num_faces}', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Face Detection & Tracking with Head Pose', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Face detection stopped.")
