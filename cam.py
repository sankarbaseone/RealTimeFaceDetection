import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS calculation variables
prev_time = 0
curr_time = 0

# 3D model points for head pose estimation
# These are the 3D coordinates of key facial landmarks in a canonical face model
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Camera internals (approximate values for a typical webcam)
def get_camera_matrix(frame_width, frame_height):
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    return camera_matrix

# Distortion coefficients (assuming no lens distortion)
dist_coeffs = np.zeros((4, 1))

def calculate_head_pose(landmarks, frame_width, frame_height):
    """Calculate head pose (yaw, pitch, roll) from facial landmarks"""
    
    # 2D image points from MediaPipe landmarks
    # Indices: 1=nose tip, 152=chin, 33=left eye left, 263=right eye right, 61=left mouth, 291=right mouth
    image_points = np.array([
        landmarks[1],      # Nose tip
        landmarks[152],    # Chin
        landmarks[33],     # Left eye left corner
        landmarks[263],    # Right eye right corner
        landmarks[61],     # Left mouth corner
        landmarks[291]     # Right mouth corner
    ], dtype=np.float64)
    
    # Get camera matrix
    camera_matrix = get_camera_matrix(frame_width, frame_height)
    
    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None, None, None
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Calculate Euler angles (yaw, pitch, roll)
    # Extract angles from rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)
    
    return pitch, yaw, roll, rotation_vector, translation_vector

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

print("Starting face detection with head pose estimation... Press 'q' to quit.")

# Start face mesh detection
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        frame_height, frame_width, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for face mesh
        results = face_mesh.process(rgb_frame)
        
        # Draw face mesh and calculate head pose
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Convert landmarks to numpy array
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    landmarks.append([x, y])
                landmarks = np.array(landmarks)
                
                # Calculate head pose
                pitch, yaw, roll, rotation_vector, translation_vector = calculate_head_pose(
                    landmarks, frame_width, frame_height
                )
                
                if pitch is not None:
                    # Get nose tip position for drawing axes
                    nose_tip = landmarks[1]
                    
                    # Draw 3D axes
                    draw_axis(frame, yaw, pitch, roll, nose_tip[0], nose_tip[1])
                    
                    # Display head pose angles
                    y_offset = 30
                    cv2.putText(frame, f'Pitch: {int(pitch)}°', (frame_width - 250, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f'Yaw: {int(yaw)}°', (frame_width - 250, y_offset + 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f'Roll: {int(roll)}°', (frame_width - 250, y_offset + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Display head orientation text
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
                    
                    cv2.putText(frame, orientation, (frame_width - 250, y_offset + 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display number of faces detected
        num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        cv2.putText(frame, f'Faces: {num_faces}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Real-Time Face Detection with Head Pose', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Face detection stopped.")
