"""
Real-Time Gesture Recognition - Complete Pipeline
Stage 1: FreiHAND model extracts keypoints
Stage 2: HaGRID classifier predicts gesture
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, r"C:\Users\Soham\Downloads\handpose_project")
from models.mobilenetv2_heatmap import MobileNetV2Heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
KEYPOINT_MODEL_PATH = r"C:\Users\Soham\Downloads\handpose_project\outputs\best_freihand_model.pth"
GESTURE_MODEL_PATH = r"C:\Users\Soham\Downloads\handpose_project\outputs\gesture_classifier.pth"

IMG_SIZE = 224
HEATMAP_SIZE = 7

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# GESTURE CLASSIFIER MODEL
# --------------------------------------------------
class GestureClassifier(nn.Module):
    def __init__(self, input_size=42, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
print("\nLoading models...")

# Load keypoint detector
print("1/2 Loading FreiHAND keypoint detector...")
keypoint_model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)
checkpoint = torch.load(KEYPOINT_MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    keypoint_model.load_state_dict(checkpoint['model_state_dict'])
else:
    keypoint_model.load_state_dict(checkpoint)
keypoint_model.to(device)
keypoint_model.eval()
print("  ✓ Keypoint model loaded")

# Load gesture classifier
print("2/2 Loading gesture classifier...")
gesture_checkpoint = torch.load(GESTURE_MODEL_PATH, map_location=device)
gesture_model = GestureClassifier(input_size=42, num_classes=6)
gesture_model.load_state_dict(gesture_checkpoint['model_state_dict'])
gesture_model.to(device)
gesture_model.eval()

# Get gesture mapping
idx_to_gesture = {v: k for k, v in gesture_checkpoint['gesture_to_idx'].items()}
print("  ✓ Gesture classifier loaded")
print(f"  ✓ Validation accuracy: {gesture_checkpoint['val_acc']:.2f}%")

print("\n✓ All models ready!\n")

# Gesture display names
GESTURE_EMOJI = {
    "two_up": "✌️ Peace Sign",
    "two_up_inverted": "✌️ Peace (Inverted)",
    "fist": "✊ Fist",
    "ok": "👌 OK Sign",
    "palm": "✋ Open Palm",
    "stop": "🛑 Stop"
}

# --------------------------------------------------
# HAND CONNECTIONS FOR VISUALIZATION
# --------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# --------------------------------------------------
# INFERENCE FUNCTIONS
# --------------------------------------------------
def extract_keypoints(frame):
    """Extract 21 keypoints from frame using FreiHAND model"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmaps = keypoint_model(img_tensor)
    
    heatmaps = heatmaps.squeeze(0).cpu().numpy()
    keypoints = []
    keypoints_normalized = []
    
    for i in range(21):
        hm = heatmaps[i]
        max_val = hm.max()
        
        if max_val < 0.2:
            keypoints.append(None)
            keypoints_normalized.extend([0.0, 0.0])
        else:
            y, x = np.unravel_index(hm.argmax(), hm.shape)
            
            # For visualization
            x_orig = int((x / HEATMAP_SIZE) * w)
            y_orig = int((y / HEATMAP_SIZE) * h)
            keypoints.append((x_orig, y_orig, max_val))
            
            # For classification
            x_norm = x / HEATMAP_SIZE
            y_norm = y / HEATMAP_SIZE
            keypoints_normalized.extend([x_norm, y_norm])
    
    return keypoints, keypoints_normalized

def recognize_gesture(keypoints_normalized):
    """Classify gesture from normalized keypoints"""
    keypoints_tensor = torch.FloatTensor(keypoints_normalized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = gesture_model(keypoints_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    gesture_idx = predicted.item()
    gesture_name = idx_to_gesture[gesture_idx]
    confidence_val = confidence.item()
    
    return gesture_name, confidence_val

def draw_keypoints(frame, keypoints):
    """Draw keypoints and skeleton"""
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        if keypoints[start_idx] and keypoints[end_idx]:
            start_pt = (keypoints[start_idx][0], keypoints[start_idx][1])
            end_pt = (keypoints[end_idx][0], keypoints[end_idx][1])
            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
    
    # Draw keypoints
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
print("=" * 60)
print("REAL-TIME GESTURE RECOGNITION")
print("=" * 60)
print("Pipeline:")
print("  1. FreiHAND model → Extract 21 keypoints")
print("  2. Gesture classifier → Recognize gesture")
print("\nSupported gestures:")
for gesture, emoji in GESTURE_EMOJI.items():
    print(f"  • {emoji}")
print("\nPress 'q' to quit, 's' to save screenshot")
print("=" * 60)
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    sys.exit(1)

gesture_buffer = deque(maxlen=5)  # Smooth over 5 frames

import time
fps_time = time.time()
screenshot_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Extract keypoints
    keypoints_viz, keypoints_norm = extract_keypoints(frame)
    
    # Recognize gesture
    gesture, confidence = recognize_gesture(keypoints_norm)
    gesture_buffer.append(gesture)
    
    # Smooth gesture prediction
    if len(gesture_buffer) > 0:
        smoothed_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
    else:
        smoothed_gesture = gesture
    
    # Draw keypoints
    frame = draw_keypoints(frame, keypoints_viz)
    
    # Calculate FPS
    fps = 1 / (time.time() - fps_time)
    fps_time = time.time()
    
    # Display info
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display gesture (large)
    gesture_text = GESTURE_EMOJI.get(smoothed_gesture, smoothed_gesture)
    cv2.putText(frame, gesture_text, (10, frame.shape[0] - 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
    
    # Display confidence
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (10, frame.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Gesture Recognition - FreiHAND + HaGRID', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_count += 1
        filename = f"gesture_screenshot_{screenshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Saved {filename}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Gesture recognition stopped.")
print("=" * 60)