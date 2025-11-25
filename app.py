from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS calculation variables
prev_time = 0
track_history = {}

def calculate_head_pose_from_bbox(bbox, frame_width, frame_height):
    """Estimate head pose from bounding box"""
    x1, y1, x2, y2 = bbox
    
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    face_width = x2 - x1
    face_height = y2 - y1
    
    frame_center_x = frame_width / 2
    yaw = ((face_center_x - frame_center_x) / frame_center_x) * 45
    
    frame_center_y = frame_height / 2
    pitch = ((face_center_y - frame_center_y) / frame_center_y) * 30
    
    aspect_ratio = face_width / face_height if face_height > 0 else 1
    roll = (aspect_ratio - 1) * 20
    
    return pitch, yaw, roll

def draw_axis(frame, yaw, pitch, roll, tdx, tdy, size=100):
    """Draw 3D axes on the face"""
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

def detect_eyes(face_roi, x, y):
    """Detect eyes in the face region"""
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
    eye_data = []
    
    for (ex, ey, ew, eh) in eyes:
        # Convert eye coordinates to frame coordinates
        eye_center_x = x + ex + ew // 2
        eye_center_y = y + ey + eh // 2
        eye_data.append({
            'x': ex, 'y': ey, 'w': ew, 'h': eh,
            'center_x': eye_center_x,
            'center_y': eye_center_y,
            'roi': face_roi[ey:ey+eh, ex:ex+ew]
        })
    
    return eye_data

def detect_pupil(eye_roi):
    """Detect pupil position in eye region using thresholding"""
    if eye_roi.size == 0:
        return None, None
    
    # Convert to grayscale if needed
    if len(eye_roi.shape) == 3:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        eye_gray = eye_roi
    
    # Apply Gaussian blur
    eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    
    # Threshold to find the darkest region (pupil)
    _, threshold_eye = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the center of the contour
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
    
    return None, None

def calculate_gaze_direction(eye_data, face_width, face_height):
    """Calculate gaze direction based on pupil position in eye"""
    if not eye_data:
        return "Unknown"
    
    gaze_directions = []
    
    for eye in eye_data:
        pupil_x, pupil_y = detect_pupil(eye['roi'])
        
        if pupil_x is not None and pupil_y is not None:
            # Calculate relative position of pupil in eye
            eye_width = eye['w']
            eye_height = eye['h']
            
            # Normalize pupil position (0 to 1)
            relative_x = pupil_x / eye_width if eye_width > 0 else 0.5
            relative_y = pupil_y / eye_height if eye_height > 0 else 0.5
            
            # Determine horizontal gaze
            if relative_x < 0.35:
                h_gaze = "Left"
            elif relative_x > 0.65:
                h_gaze = "Right"
            else:
                h_gaze = "Center"
            
            # Determine vertical gaze
            if relative_y < 0.35:
                v_gaze = "Up"
            elif relative_y > 0.65:
                v_gaze = "Down"
            else:
                v_gaze = ""
            
            # Combine directions
            if h_gaze == "Center" and v_gaze == "":
                gaze = "Center"
            elif v_gaze:
                gaze = f"{h_gaze} & {v_gaze}"
            else:
                gaze = h_gaze
            
            gaze_directions.append(gaze)
    
    # Return most common gaze direction
    if gaze_directions:
        return max(set(gaze_directions), key=gaze_directions.count)
    
    return "Center"

def generate_frames():
    """Generate frames from camera with face detection"""
    global prev_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process detected faces
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Calculate face center
                face_center_x = int((x1 + x2) / 2)
                face_center_y = int((y1 + y2) / 2)
                
                # Draw bounding box
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
                
                # Detect eyes in face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]
                eye_data = detect_eyes(face_roi, x, y)
                
                # Draw eye bounding boxes and detect gaze
                for eye in eye_data:
                    ex, ey, ew, eh = eye['x'], eye['y'], eye['w'], eye['h']
                    # Draw eye rectangle on frame
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                    
                    # Detect and draw pupil
                    pupil_x, pupil_y = detect_pupil(eye['roi'])
                    if pupil_x is not None and pupil_y is not None:
                        # Draw pupil center
                        pupil_frame_x = x + ex + pupil_x
                        pupil_frame_y = y + ey + pupil_y
                        cv2.circle(frame, (pupil_frame_x, pupil_frame_y), 3, (0, 255, 255), -1)
                
                # Calculate gaze direction
                gaze_direction = calculate_gaze_direction(eye_data, w, h)
                
                # Display head pose angles (for first face only)
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
                    
                    # Display gaze direction
                    cv2.putText(frame, f'Gaze: {gaze_direction}', (frame_width - 250, y_offset + 145), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
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
        
        # Display number of faces
        num_faces = len(faces)
        cv2.putText(frame, f'Faces: {num_faces}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
