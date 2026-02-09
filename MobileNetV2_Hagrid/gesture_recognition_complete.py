"""
Complete Gesture Recognition System
- Uses MediaPipe for hand detection (no training needed)
- Simple gesture classifier for 6 gestures
- Real-time inference
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --------------------------------------------------
# GESTURE DEFINITIONS (6 classes)
# --------------------------------------------------
GESTURES = {
    0: "FIST",
    1: "PALM",
    2: "THUMBS_UP",
    3: "PEACE",
    4: "OK",
    5: "POINTING"
}


class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Smoothing buffer
        self.gesture_buffer = deque(maxlen=5)
    
    def get_finger_states(self, landmarks):
        """
        Returns which fingers are extended [thumb, index, middle, ring, pinky]
        True = extended, False = folded
        """
        fingers = []
        
        # Thumb (compare x-coordinate)
        if landmarks[4].x < landmarks[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
        
        # Other fingers (compare y-coordinate)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(True)
            else:
                fingers.append(False)
        
        return fingers
    
    def classify_gesture(self, fingers):
        """
        Classify gesture based on finger states
        fingers = [thumb, index, middle, ring, pinky]
        """
        # FIST - all fingers closed
        if not any(fingers):
            return 0
        
        # PALM - all fingers open
        if all(fingers):
            return 1
        
        # THUMBS_UP - only thumb extended
        if fingers == [True, False, False, False, False]:
            return 2
        
        # PEACE - index and middle extended
        if fingers == [False, True, True, False, False]:
            return 3
        
        # OK - thumb and index touching (approximation: thumb + middle/ring/pinky)
        if fingers[0] and fingers[2] and not fingers[1]:
            return 4
        
        # POINTING - only index extended
        if fingers == [False, True, False, False, False]:
            return 5
        
        # Default to PALM if unclear
        return 1
    
    def get_smoothed_gesture(self, gesture_id):
        """Smooth gestures over time to reduce jitter"""
        self.gesture_buffer.append(gesture_id)
        
        # Return most common gesture in buffer
        if len(self.gesture_buffer) > 0:
            return max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        return gesture_id
    
    def process_frame(self, frame):
        """
        Process a single frame and return annotated frame with gesture
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        gesture_name = "NO HAND"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get finger states
                fingers = self.get_finger_states(hand_landmarks.landmark)
                
                # Classify gesture
                gesture_id = self.classify_gesture(fingers)
                gesture_id = self.get_smoothed_gesture(gesture_id)
                gesture_name = GESTURES[gesture_id]
        
        # Display gesture name
        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )
        
        return frame, gesture_name


# --------------------------------------------------
# REAL-TIME DEMO
# --------------------------------------------------
def main():
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)
    
    print("Starting gesture recognition...")
    print("Gestures supported:", list(GESTURES.values()))
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        annotated_frame, gesture = recognizer.process_frame(frame)
        
        # Show frame
        cv2.imshow('Gesture Recognition', annotated_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture recognition stopped.")


if __name__ == "__main__":
    main()