#!/usr/bin/env python3
"""
Visualize IPN Hand Dataset
Plot distributions and visualize gesture sequences
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse


# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


def plot_distribution(data_dir):
    """
    Plot the distribution of sequences per gesture
    
    Args:
        data_dir: Directory with .npy files
    """
    data_path = Path(data_dir)
    
    # Load all gesture data
    gesture_counts = {}
    for npy_file in sorted(data_path.glob("*.npy")):
        gesture_name = npy_file.stem
        data = np.load(npy_file)
        gesture_counts[gesture_name] = len(data)
    
    if not gesture_counts:
        print("❌ No .npy files found in directory")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gestures = list(gesture_counts.keys())
    counts = list(gesture_counts.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(gestures)))
    
    bars = ax.bar(gestures, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Gesture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax.set_title('IPN Hand Dataset Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and show
    output_file = data_path / 'distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Distribution plot saved to: {output_file}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    total = sum(counts)
    print(f"\nTotal sequences: {total:,}")
    print(f"Number of gestures: {len(gestures)}")
    print(f"Average per gesture: {total/len(gestures):.0f}")
    print(f"Min: {min(counts):,} ({gestures[counts.index(min(counts))]})")
    print(f"Max: {max(counts):,} ({gestures[counts.index(max(counts))]})")


def visualize_sequence(data_dir, gesture_name, sequence_idx=0):
    """
    Visualize a single gesture sequence as animation
    
    Args:
        data_dir: Directory with .npy files
        gesture_name: Name of the gesture to visualize
        sequence_idx: Index of sequence to visualize
    """
    data_path = Path(data_dir)
    npy_file = data_path / f"{gesture_name}.npy"
    
    if not npy_file.exists():
        print(f"❌ Gesture file not found: {npy_file}")
        return
    
    # Load data
    data = np.load(npy_file)
    
    if sequence_idx >= len(data):
        print(f"❌ Sequence index {sequence_idx} out of range (max: {len(data)-1})")
        return
    
    sequence = data[sequence_idx]  # Shape: (seq_len, 42)
    
    # Reshape to (seq_len, 21, 2)
    landmarks = sequence.reshape(-1, 21, 2)
    
    print(f"\n📊 Visualizing: {gesture_name}, sequence {sequence_idx}")
    print(f"   Sequence length: {len(landmarks)} frames")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame_idx):
        ax.clear()
        
        # Get current frame landmarks
        points = landmarks[frame_idx]
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=3)
        
        # Plot connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            x_coords = [points[start_idx, 0], points[end_idx, 0]]
            y_coords = [points[start_idx, 1], points[end_idx, 1]]
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, zorder=2)
        
        # Add landmark numbers
        for i, point in enumerate(points):
            ax.annotate(str(i), (point[0], point[1]), 
                       fontsize=8, ha='center', va='center',
                       color='white', weight='bold',
                       bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.5))
        
        # Set limits and labels
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{gesture_name} - Frame {frame_idx+1}/{len(landmarks)}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(landmarks), 
                        interval=100, repeat=True)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize gesture dataset")
    parser.add_argument('--data_dir', type=str, default='ipn_processed',
                       help='Directory with .npy files')
    parser.add_argument('--action', type=str, choices=['distribution', 'sequence'],
                       default='distribution',
                       help='What to visualize')
    parser.add_argument('--gesture', type=str, default='open_palm',
                       help='Gesture to visualize (for sequence action)')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Sequence index to visualize')
    
    args = parser.parse_args()
    
    if args.action == 'distribution':
        plot_distribution(args.data_dir)
    elif args.action == 'sequence':
        visualize_sequence(args.data_dir, args.gesture, args.sequence_idx)


if __name__ == "__main__":
    main()