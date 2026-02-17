#!/usr/bin/env python3
"""
Data Augmentation for Gesture Sequences
Applies various augmentation techniques to increase dataset size
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class GestureAugmenter:
    """Augment gesture sequence data"""
    
    def __init__(self):
        """Initialize augmenter"""
        pass
    
    def add_gaussian_noise(self, sequence, sigma=0.01):
        """
        Add Gaussian noise to keypoints
        
        Args:
            sequence: Array of shape (seq_len, 42)
            sigma: Standard deviation of noise
            
        Returns:
            Augmented sequence
        """
        noise = np.random.normal(0, sigma, sequence.shape)
        return sequence + noise
    
    def horizontal_flip(self, sequence):
        """
        Flip hand horizontally (mirror)
        
        Args:
            sequence: Array of shape (seq_len, 42)
            
        Returns:
            Flipped sequence
        """
        # Reshape to (seq_len, 21, 2)
        landmarks = sequence.reshape(-1, 21, 2)
        
        # Flip x coordinates
        landmarks[:, :, 0] = -landmarks[:, :, 0]
        
        # Flatten back
        return landmarks.reshape(-1, 42)
    
    def random_rotation(self, sequence, max_angle=10):
        """
        Randomly rotate hand landmarks
        
        Args:
            sequence: Array of shape (seq_len, 42)
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated sequence
        """
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Reshape to (seq_len, 21, 2)
        landmarks = sequence.reshape(-1, 21, 2)
        
        # Apply rotation
        rotated = np.zeros_like(landmarks)
        for i in range(landmarks.shape[0]):
            rotated[i] = landmarks[i] @ rotation_matrix.T
        
        # Flatten back
        return rotated.reshape(-1, 42)
    
    def random_scale(self, sequence, scale_range=(0.9, 1.1)):
        """
        Randomly scale hand size
        
        Args:
            sequence: Array of shape (seq_len, 42)
            scale_range: (min_scale, max_scale)
            
        Returns:
            Scaled sequence
        """
        scale = np.random.uniform(*scale_range)
        return sequence * scale
    
    def time_warp(self, sequence, warp_factor=1.2):
        """
        Speed up or slow down the sequence
        
        Args:
            sequence: Array of shape (seq_len, 42)
            warp_factor: >1 for speedup, <1 for slowdown
            
        Returns:
            Time-warped sequence of same length
        """
        seq_len = len(sequence)
        
        # Create new time indices
        old_indices = np.arange(seq_len)
        new_indices = np.linspace(0, seq_len - 1, int(seq_len * warp_factor))
        
        # Interpolate
        warped = np.zeros((seq_len, 42))
        for i in range(42):
            warped[:, i] = np.interp(
                np.linspace(0, len(new_indices) - 1, seq_len),
                np.arange(len(new_indices)),
                np.interp(new_indices, old_indices, sequence[:, i])
            )
        
        return warped
    
    def augment_sequence(self, sequence, augmentation_type):
        """
        Apply specific augmentation
        
        Args:
            sequence: Array of shape (seq_len, 42)
            augmentation_type: Type of augmentation
            
        Returns:
            Augmented sequence
        """
        if augmentation_type == 'noise_small':
            return self.add_gaussian_noise(sequence, sigma=0.01)
        elif augmentation_type == 'noise_medium':
            return self.add_gaussian_noise(sequence, sigma=0.02)
        elif augmentation_type == 'flip':
            return self.horizontal_flip(sequence)
        elif augmentation_type == 'rotation':
            return self.random_rotation(sequence, max_angle=10)
        elif augmentation_type == 'scale':
            return self.random_scale(sequence, scale_range=(0.9, 1.1))
        elif augmentation_type == 'time_warp_fast':
            return self.time_warp(sequence, warp_factor=1.2)
        elif augmentation_type == 'time_warp_slow':
            return self.time_warp(sequence, warp_factor=0.8)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def augment_dataset(self, input_dir, output_dir, factor=3):
        """
        Augment entire dataset
        
        Args:
            input_dir: Directory with original .npy files
            output_dir: Output directory for augmented data
            factor: Augmentation factor (total size = original * factor)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Augmentation strategies
        augmentation_types = [
            'noise_small',
            'noise_medium', 
            'flip',
            'rotation',
            'scale',
            'time_warp_fast',
            'time_warp_slow'
        ]
        
        print("\n" + "="*80)
        print("AUGMENTING DATASET")
        print("="*80)
        print(f"\nAugmentation factor: {factor}x")
        print(f"Augmentation types: {len(augmentation_types)}")
        
        # Process each gesture
        for npy_file in sorted(input_path.glob("*.npy")):
            gesture_name = npy_file.stem
            
            print(f"\n📊 Processing {gesture_name}")
            
            # Load original data
            original_data = np.load(npy_file)
            print(f"  Original sequences: {len(original_data)}")
            
            # Start with original data
            augmented_data = [original_data]
            
            # Calculate how many augmented versions we need
            num_augmentations = factor - 1
            
            # Augment
            for aug_idx in range(num_augmentations):
                # Cycle through augmentation types
                aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                
                augmented_sequences = []
                for sequence in tqdm(original_data, desc=f"  Aug {aug_idx+1}/{num_augmentations} ({aug_type})"):
                    augmented_seq = self.augment_sequence(sequence, aug_type)
                    augmented_sequences.append(augmented_seq)
                
                augmented_data.append(np.array(augmented_sequences))
            
            # Combine all augmented data
            combined_data = np.concatenate(augmented_data, axis=0)
            
            # Shuffle
            np.random.shuffle(combined_data)
            
            # Save
            output_file = output_path / f"{gesture_name}.npy"
            np.save(output_file, combined_data)
            
            print(f"  ✓ Augmented sequences: {len(combined_data)}")
            print(f"  ✓ Saved to: {output_file.name}")
        
        # Update metadata
        if (input_path / 'metadata.json').exists():
            with open(input_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            metadata['augmented'] = True
            metadata['augmentation_factor'] = factor
            metadata['augmentation_types'] = augmentation_types
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print("\n" + "="*80)
        print("AUGMENTATION COMPLETE")
        print("="*80)
        print(f"\nAugmented dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment gesture dataset")
    parser.add_argument('--data_dir', type=str, default='ipn_processed',
                       help='Input directory with .npy files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: data_dir/augmented)')
    parser.add_argument('--factor', type=int, default=3,
                       help='Augmentation factor (default: 3x)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_dir) / 'augmented')
    
    # Augment dataset
    augmenter = GestureAugmenter()
    augmenter.augment_dataset(args.data_dir, args.output_dir, args.factor)


if __name__ == "__main__":
    main()