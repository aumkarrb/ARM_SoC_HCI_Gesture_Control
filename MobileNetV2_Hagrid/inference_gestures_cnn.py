"""
Real-Time Gesture Recognition - CNN Approach
Direct classification from images (no keypoints needed)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = r"C:\Users\Soham\Downloads\handpose_project\outputs\gesture_cnn.pth"
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("\nLoading gesture classifier...")

# Create model architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 6)  # 6 gestures
)

# Load weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Get gesture mapping
idx_to_gesture = {v: k for k, v in checkpoint['gesture_to_idx'].items()}
print(f"✓ Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)\n")

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
# PREPROCESSING
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
def recognize_gesture(frame):
    """Classify gesture from frame"""
    # Preprocess
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    gesture_idx = predicted.item()
    gesture_name = idx_to_gesture[gesture_idx]
    confidence_val = confidence.item()
    
    return gesture_name, confidence_val

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
print("=" * 60)
print("REAL-TIME GESTURE RECOGNITION")
print("=" * 60)
print("Supported gestures:")
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
    
    # Recognize gesture
    gesture, confidence = recognize_gesture(frame)
    gesture_buffer.append(gesture)
    
    # Smooth gesture prediction
    if len(gesture_buffer) > 0:
        smoothed_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
    else:
        smoothed_gesture = gesture
    
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
    
    cv2.imshow('Gesture Recognition - CNN', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        screenshot_count += 1
        filename = f"gesture_cnn_screenshot_{screenshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Saved {filename}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Gesture recognition stopped.")
print("=" * 60)