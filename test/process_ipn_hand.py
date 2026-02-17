#!/usr/bin/env python3
"""
IPN Hand Dataset Processor
Extracts hand keypoints from IPN Hand videos using MediaPipe
"""
print("SCRIPT STARTED")

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class IPNHandProcessor:
    """Process IPN Hand videos and extract normalized keypoints"""
    
    # IPN Hand gesture class mapping to our 8 target gestures
    GESTURE_MAPPING = {
        'D01': 'index_point',      # Pointing with one finger
        'D02': 'index_point',      # Pointing with two fingers
        'D03': 'index_point',      # Click with one finger
        'D04': 'pinch',            # Click with two fingers
        'D05': 'thumb_up',         # Throw up
        'D06': 'thumb_down',       # Throw down
        'D07': 'swipe_left',       # Throw left
        'D08': 'swipe_right',      # Throw right
        'D09': 'open_palm',        # Open hand
        'D10': 'closed_fist',      # Grab with hand
        'D11': 'open_palm',        # Expand with hand
        'D12': 'pinch',            # Pinch with hand
        'D13': 'closed_fist',      # Move with hand
    }
    
    def __init__(self, sequence_length=10):
        """
        Initialize the processor
        
        Args:
            sequence_length: Number of frames per sequence
        """
        self.sequence_length = sequence_length
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to be translation and scale invariant
        
        Args:
            landmarks: Array of shape (21, 2) containing (x, y) coordinates
            
        Returns:
            Normalized landmarks of shape (21, 2)
        """
        # Center around wrist (landmark 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Scale based on hand size (distance from wrist to middle finger tip)
        middle_tip = centered[12]  # Middle finger tip
        scale = np.linalg.norm(middle_tip) + 1e-6
        normalized = centered / scale
        
        return normalized
    
    def extract_keypoints_from_frame(self, frame):
        """
        Extract hand keypoints from a single frame
        
        Args:
            frame: BGR image
            
        Returns:
            Normalized keypoints of shape (42,) or None if no hand detected
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract (x, y) coordinates
            landmarks = np.array([
                [lm.x, lm.y] for lm in hand_landmarks.landmark
            ])  # Shape: (21, 2)
            
            # Normalize
            normalized = self.normalize_landmarks(landmarks)
            
            # Flatten to (42,)
            return normalized.flatten()
        
        return None
    
    def process_video(self, video_path):
        """
        Process a single video file and extract sequences
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of sequences, each of shape (sequence_length, 42)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return []
        
        keypoints_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints = self.extract_keypoints_from_frame(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
        
        cap.release()
        
        # Create sequences
        sequences = []
        if len(keypoints_list) >= self.sequence_length:
            for i in range(len(keypoints_list) - self.sequence_length + 1):
                sequence = np.array(keypoints_list[i:i + self.sequence_length])
                sequences.append(sequence)
        
        return sequences
    
    def process_dataset(self, ipn_dir, output_dir, max_videos_per_class=None):
        """
        Process entire IPN Hand dataset
        
        Args:
            ipn_dir: Path to IPN frames directory
            output_dir: Output directory for processed data
            max_videos_per_class: Maximum videos to process per class (for testing)
            
        Returns:
            Dictionary with statistics
        """
        ipn_path = Path(ipn_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_videos': 0,
            'total_sequences': 0,
            'gesture_counts': {},
            'failed_videos': []
        }
        
        print("\n" + "="*80)
        print("PROCESSING IPN HAND DATASET")
        print("="*80)
        
        # Process each IPN gesture class
        for ipn_class in sorted(ipn_path.iterdir()):
            if not ipn_class.is_dir():
                continue
            ipn_class_name = ipn_class.name
            if ipn_class_name not in self.GESTURE_MAPPING:
                print(f"⚠ Skipping unknown class: {ipn_class_name}")
                continue

            
            ipn_class_name = ipn_class.name
            if ipn_class_name not in self.GESTURE_MAPPING:
                print(f"⚠ Skipping unknown class: {ipn_class_name}")
                continue
            
            target_gesture = self.GESTURE_MAPPING[ipn_class_name]
            
            print(f"\n📹 Processing {ipn_class_name} → {target_gesture}")
            
            # Get all video files
            video_files = sorted(ipn_class.glob("*.avi"))
            if max_videos_per_class:
                video_files = video_files[:max_videos_per_class]
            
            gesture_sequences = []
            
            for video_file in tqdm(video_files, desc=f"  {target_gesture}"):
                try:
                    sequences = self.process_video(video_file)
                    gesture_sequences.extend(sequences)
                    stats['total_videos'] += 1
                except Exception as e:
                    stats['failed_videos'].append((str(video_file), str(e)))
            
            # Save sequences for this gesture
            if gesture_sequences:
                gesture_array = np.array(gesture_sequences)
                output_file = output_path / f"{target_gesture}.npy"
                
                # Append if file exists
                if output_file.exists():
                    existing = np.load(output_file)
                    gesture_array = np.concatenate([existing, gesture_array], axis=0)
                
                np.save(output_file, gesture_array)
                
                stats['gesture_counts'][target_gesture] = len(gesture_array)
                stats['total_sequences'] += len(gesture_sequences)
                
                print(f"  ✓ Saved {len(gesture_array)} sequences to {output_file.name}")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'num_features': 42,
            'gesture_mapping': self.GESTURE_MAPPING,
            'stats': stats
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return stats
    
    def create_balanced_dataset(self, input_dir, output_dir):
        """
        Create a balanced dataset by undersampling
        
        Args:
            input_dir: Directory with processed .npy files
            output_dir: Output directory for balanced dataset
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("CREATING BALANCED DATASET")
        print("="*80)
        
        # Load all gesture data
        gesture_data = {}
        for npy_file in input_path.glob("*.npy"):
            gesture_name = npy_file.stem
            data = np.load(npy_file)
            gesture_data[gesture_name] = data
            print(f"  {gesture_name}: {len(data)} sequences")
        
        # Find minimum count
        min_count = min(len(data) for data in gesture_data.values())
        print(f"\n📊 Minimum count: {min_count}")
        print(f"   Balancing all classes to {min_count} sequences")
        
        # Undersample and save
        for gesture_name, data in gesture_data.items():
            # Randomly sample
            indices = np.random.choice(len(data), min_count, replace=False)
            balanced_data = data[indices]
            
            # Save
            output_file = output_path / f"{gesture_name}.npy"
            np.save(output_file, balanced_data)
            print(f"  ✓ Saved {gesture_name}: {len(balanced_data)} sequences")
        
        # Copy metadata
        if (input_path / 'metadata.json').exists():
            with open(input_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            metadata['balanced'] = True
            metadata['samples_per_class'] = min_count
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Balanced dataset saved to: {output_path}")
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    parser = argparse.ArgumentParser(description="Process IPN Hand dataset")
    parser.add_argument('--ipn_dir', type=str, default='ipn_hand_dataset/frames',
                       help='Path to IPN frames directory')
    parser.add_argument('--output_dir', type=str, default='ipn_processed',
                       help='Output directory for processed data')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Number of frames per sequence')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum videos per class (for testing)')
    parser.add_argument('--balance', action='store_true',
                       help='Create balanced dataset after processing')
    
    args = parser.parse_args()
    print("ARGS:", args)
    
    print("Folders found:")
    for item in ipn_path.iterdir():
        print(repr(item.name))

    
    # Process dataset
    processor = IPNHandProcessor(sequence_length=args.sequence_length)
    stats = processor.process_dataset(
        args.ipn_dir, 
        args.output_dir,
        max_videos_per_class=args.max_videos
    )
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  Total videos processed: {stats['total_videos']}")
    print(f"  Total sequences created: {stats['total_sequences']}")
    print(f"\n  Sequences per gesture:")
    for gesture, count in sorted(stats['gesture_counts'].items()):
        print(f"    {gesture:15s}: {count:5d}")
    
    if stats['failed_videos']:
        print(f"\n  ⚠ Failed videos: {len(stats['failed_videos'])}")
    
    # Create balanced dataset if requested
    if args.balance:
        balanced_dir = Path(args.output_dir) / 'balanced'
        processor.create_balanced_dataset(args.output_dir, balanced_dir)


if __name__ == "__main__":
    main()