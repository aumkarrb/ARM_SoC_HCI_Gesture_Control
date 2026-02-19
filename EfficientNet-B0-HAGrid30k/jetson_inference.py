#!/usr/bin/env python3
"""
Real-time Gesture Recognition on Jetson Nano
Two-stage pipeline: trt_pose + TensorRT gesture classifier
RUN THIS ON JETSON NANO
"""

import cv2
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import json
from collections import deque
import time
import argparse
from pathlib import Path

# trt_pose imports (install on Jetson: https://github.com/NVIDIA-AI-IOT/trt_pose)
from trt_pose.coco import coco_category_to_topology
from trt_pose.models import resnet18_baseline_att
import trt_pose.models
import torch2trt


# Hand topology for trt_pose
HAND_TOPOLOGY = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],  # Index
    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
    [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
    [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
]


class TRTGestureClassifier:
    """TensorRT gesture classifier"""
    
    def __init__(self, engine_path, metadata_path):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path: Path to TensorRT engine file
            metadata_path: Path to metadata JSON
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.input_shape = tuple(self.metadata['input_shape'])
        self.num_classes = self.metadata['num_classes']
        
        # Create gesture index to name mapping
        gesture_to_idx = self.metadata['gesture_to_idx']
        self.idx_to_gesture = {idx: name for name, idx in gesture_to_idx.items()}
        
        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty((1, self.num_classes), dtype=np.float32)
        
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        self.stream = cuda.Stream()
    
    def predict(self, sequence):
        """
        Predict gesture from sequence
        
        Args:
            sequence: Numpy array of shape (seq_len, 42)
            
        Returns:
            (gesture_name, confidence)
        """
        # Prepare input
        input_data = sequence.reshape(self.input_shape).astype(np.float32)
        np.copyto(self.h_input, input_data)
        
        # Transfer to device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Run inference
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        # Transfer back
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        # Get prediction
        predicted_idx = np.argmax(self.h_output[0])
        confidence = np.max(self.h_output[0])
        gesture_name = self.idx_to_gesture[predicted_idx]
        
        return gesture_name, confidence


class HandPoseDetector:
    """Hand pose detection using trt_pose"""
    
    def __init__(self, model_path='hand_pose_resnet18_att_244_244.pth'):
        """
        Initialize hand pose detector
        
        Args:
            model_path: Path to trt_pose model
        """
        # Note: Download model from:
        # https://github.com/NVIDIA-AI-IOT/trt_pose
        
        # Load model
        num_parts = 21  # Hand landmarks
        num_links = len(HAND_TOPOLOGY)
        
        self.model = resnet18_baseline_att(num_parts, 2 * num_links)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda().eval()
        
        # Convert to TensorRT
        data = torch.zeros((1, 3, 224, 224)).cuda()
        self.model_trt = torch2trt.torch2trt(
            self.model,
            [data],
            fp16_mode=True,
            max_workspace_size=1<<25
        )
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    
    def preprocess(self, image):
        """Preprocess image for model"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = torch.from_numpy(image).cuda().float()
        image = image.permute(2, 0, 1) / 255.0
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return image[None, ...]
    
    def extract_keypoints(self, image):
        """
        Extract hand keypoints from image
        
        Args:
            image: BGR image
            
        Returns:
            Normalized keypoints of shape (42,) or None if no hand detected
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        with torch.no_grad():
            cmap, paf = self.model_trt(input_tensor)
        
        # Parse keypoints (simplified version)
        # In production, use trt_pose's parse_objects for better accuracy
        cmap = cmap[0].cpu().numpy()
        
        # Find peaks in confidence maps
        keypoints = []
        for i in range(21):
            confidence_map = cmap[i]
            y, x = np.unravel_index(np.argmax(confidence_map), confidence_map.shape)
            
            # Normalize to [-1, 1]
            x_norm = (x / confidence_map.shape[1]) * 2 - 1
            y_norm = (y / confidence_map.shape[0]) * 2 - 1
            
            keypoints.append([x_norm, y_norm])
        
        keypoints = np.array(keypoints)
        
        # Check if hand is detected (simple threshold)
        if np.max(cmap) < 0.3:
            return None
        
        # Normalize (translation + scale invariant)
        wrist = keypoints[0]
        centered = keypoints - wrist
        
        middle_tip = centered[12]
        scale = np.linalg.norm(middle_tip) + 1e-6
        normalized = centered / scale
        
        return normalized.flatten()  # Shape: (42,)


class GestureRecognitionPipeline:
    """Complete gesture recognition pipeline"""
    
    def __init__(self, gesture_engine_path, gesture_metadata_path, 
                 hand_model_path, sequence_length=10, confidence_threshold=0.7):
        """
        Initialize pipeline
        
        Args:
            gesture_engine_path: Path to gesture classifier TensorRT engine
            gesture_metadata_path: Path to metadata JSON
            hand_model_path: Path to trt_pose hand model
            sequence_length: Number of frames per sequence
            confidence_threshold: Minimum confidence for prediction
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        print("🤖 Loading gesture classifier...")
        self.gesture_classifier = TRTGestureClassifier(
            gesture_engine_path,
            gesture_metadata_path
        )
        
        print("👋 Loading hand pose detector...")
        self.hand_detector = HandPoseDetector(hand_model_path)
        
        # Sequence buffer
        self.keypoint_buffer = deque(maxlen=sequence_length)
        
        print("✓ Pipeline initialized")
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: BGR image
            
        Returns:
            (gesture_name, confidence) or (None, 0) if no prediction
        """
        # Extract keypoints
        keypoints = self.hand_detector.extract_keypoints(frame)
        
        if keypoints is None:
            # No hand detected
            self.keypoint_buffer.clear()
            return None, 0.0
        
        # Add to buffer
        self.keypoint_buffer.append(keypoints)
        
        # Check if buffer is full
        if len(self.keypoint_buffer) < self.sequence_length:
            return None, 0.0
        
        # Create sequence
        sequence = np.array(self.keypoint_buffer)
        
        # Predict gesture
        gesture, confidence = self.gesture_classifier.predict(sequence)
        
        if confidence < self.confidence_threshold:
            return None, confidence
        
        return gesture, confidence


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture recognition on Jetson")
    parser.add_argument('--engine', type=str, required=True,
                       help='Path to TensorRT gesture classifier engine')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata JSON (default: engine.json)')
    parser.add_argument('--hand_model', type=str, 
                       default='hand_pose_resnet18_att_244_244.pth',
                       help='Path to trt_pose hand model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Set metadata path
    if args.metadata is None:
        args.metadata = Path(args.engine).with_suffix('.json')
    
    print("="*80)
    print("JETSON NANO GESTURE RECOGNITION")
    print("="*80)
    
    # Initialize pipeline
    pipeline = GestureRecognitionPipeline(
        args.engine,
        args.metadata,
        args.hand_model,
        confidence_threshold=args.confidence
    )
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n📹 Camera opened: {args.camera}")
    print("Press 'q' to quit\n")
    
    # FPS calculation
    fps_deque = deque(maxlen=30)
    
    current_gesture = "No gesture"
    current_confidence = 0.0
    
    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        gesture, confidence = pipeline.process_frame(frame)
        
        if gesture is not None:
            current_gesture = gesture
            current_confidence = confidence
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        fps_deque.append(fps)
        avg_fps = np.mean(fps_deque)
        
        # Draw overlay
        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Stopped")


if __name__ == "__main__":
    main()