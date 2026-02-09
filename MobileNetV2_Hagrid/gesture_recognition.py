"""
Hand Gesture Recognition Using Keypoint Detection
Recognizes 6 common gestures based on detected keypoints
"""

import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))
from models.mobilenetv2_heatmap import MobileNetV2Heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "outputs/best_freihand_model.pth"
IMG_SIZE = 224
HEATMAP_SIZE = 7
CONFIDENCE_THRESHOLD = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# GESTURE DEFINITIONS
# --------------------------------------------------
GESTURES = {
    "OPEN_PALM": "✋ Open Palm",
    "FIST": "✊ Fist",
    "PEACE": "✌️ Peace Sign",
    "THUMBS_UP": "👍 Thumbs Up",
    "POINTING": "☝️ Pointing",
    "OK": "👌 OK Sign",
    "UNKNOWN": "❓ Unknown"
}

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading model...")
model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("✓ Model loaded\n")

# --------------------------------------------------
# HAND CONNECTIONS
# --------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# --------------------------------------------------
# KEYPOINT DETECTION
# --------------------------------------------------
def detect_hand_keypoints(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmaps = model(img_tensor)
    
    heatmaps = heatmaps.squeeze(0).cpu().numpy()
    keypoints = []
    
    for i in range(21):
        hm = heatmaps[i]
        max_val = hm.max()
        if max_val < CONFIDENCE_THRESHOLD:
            keypoints.append(None)
            continue
        y, x = np.unravel_index(hm.argmax(), hm.shape)
        x_orig = int((x / HEATMAP_SIZE) * w)
        y_orig = int((y / HEATMAP_SIZE) * h)
        keypoints.append((x_orig, y_orig, max_val))
    
    return keypoints

# --------------------------------------------------
# GESTURE RECOGNITION
# --------------------------------------------------
def get_finger_states(keypoints):
    """
    Returns which fingers are extended
    [thumb, index, middle, ring, pinky]
    """
    if keypoints[0] is None:  # No wrist detected
        return None
    
    fingers = []
    
    # Thumb - check if tip is higher than base
    if keypoints[4] and keypoints[2]:
        fingers.append(keypoints[4][1] < keypoints[2][1])
    else:
        fingers.append(False)
    
    # Other fingers - check if tip is higher than middle joint
    finger_tips = [8, 12, 16, 20]
    finger_mids = [6, 10, 14, 18]
    
    for tip_idx, mid_idx in zip(finger_tips, finger_mids):
        if keypoints[tip_idx] and keypoints[mid_idx]:
            fingers.append(keypoints[tip_idx][1] < keypoints[mid_idx][1])
        else:
            fingers.append(False)
    
    return fingers

def recognize_gesture(keypoints):
    """Classify gesture based on keypoint positions"""
    fingers = get_finger_states(keypoints)
    
    if fingers is None:
        return "UNKNOWN"
    
    extended_count = sum(fingers)
    
    # FIST - no fingers extended
    if extended_count == 0:
        return "FIST"
    
    # OPEN_PALM - all fingers extended
    if extended_count == 5:
        return "OPEN_PALM"
    
    # PEACE - index and middle extended
    if fingers == [False, True, True, False, False]:
        return "PEACE"
    
    # POINTING - only index extended
    if fingers == [False, True, False, False, False]:
        return "POINTING"
    
    # THUMBS_UP - only thumb extended
    if fingers == [True, False, False, False, False]:
        return "THUMBS_UP"
    
    # OK - thumb and index in circle (approximation)
    if fingers[0] and extended_count >= 3:
        return "OK"
    
    return "UNKNOWN"

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
def draw_keypoints(frame, keypoints):
    for start_idx, end_idx in HAND_CONNECTIONS:
        if keypoints[start_idx] and keypoints[end_idx]:
            start_pt = (keypoints[start_idx][0], keypoints[start_idx][1])
            end_pt = (keypoints[end_idx][0], keypoints[end_idx][1])
            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
    
    for i, kp in enumerate(keypoints):
        if kp:
            x, y, conf = kp
            if i == 0:
                color = (0, 0, 255)
            elif i in [4, 8, 12, 16, 20]:
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 1)
    
    return frame

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
print("Starting Gesture Recognition...")
print("Supported gestures:", list(GESTURES.values()))
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)
gesture_buffer = deque(maxlen=10)  # Smooth gestures over 10 frames

import time
fps_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    keypoints = detect_hand_keypoints(frame)
    frame = draw_keypoints(frame, keypoints)
    
    # Recognize gesture
    gesture = recognize_gesture(keypoints)
    gesture_buffer.append(gesture)
    
    # Get most common gesture in buffer (smoothing)
    if len(gesture_buffer) > 0:
        smoothed_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
    else:
        smoothed_gesture = "UNKNOWN"
    
    # Display gesture
    gesture_text = GESTURES[smoothed_gesture]
    
    # Calculate FPS
    fps = 1 / (time.time() - fps_time)
    fps_time = time.time()
    
    # Draw UI
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw gesture in large text
    cv2.putText(frame, gesture_text, (10, frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Gesture recognition stopped.")