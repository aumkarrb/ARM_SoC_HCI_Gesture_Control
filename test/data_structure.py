#!/usr/bin/env python3
"""
Quick diagnostic script to check IPN Hand dataset structure
"""
import os
from pathlib import Path
import argparse

def check_structure(ipn_dir):
    """Check the actual structure of the IPN dataset"""
    ipn_path = Path(ipn_dir)
    
    print("\n" + "="*80)
    print(f"CHECKING DATASET STRUCTURE: {ipn_path}")
    print("="*80)
    
    if not ipn_path.exists():
        print(f"❌ ERROR: Directory does not exist: {ipn_path}")
        return
    
    # List all items in the directory
    items = list(ipn_path.iterdir())
    
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]
    
    print(f"\n📁 Total folders found: {len(folders)}")
    print(f"📄 Total files found: {len(files)}")
    
    # Check folders
    print("\n" + "-"*80)
    print("FOLDERS:")
    print("-"*80)
    for folder in sorted(folders):
        # Count files in each folder
        avi_files = list(folder.glob("*.avi"))
        all_files = list(folder.glob("*"))
        print(f"  {folder.name:10s} - {len(avi_files):4d} AVI files, {len(all_files):4d} total files")
    
    # Check AVI files at root level
    print("\n" + "-"*80)
    print("AVI FILES AT ROOT LEVEL:")
    print("-"*80)
    avi_files_root = [f for f in files if f.suffix == '.avi']
    print(f"  Found {len(avi_files_root)} AVI files at root level")
    if avi_files_root:
        print(f"  Examples:")
        for f in avi_files_root[:5]:
            print(f"    - {f.name}")
    
    # Check expected structure (D01-D13 folders)
    print("\n" + "-"*80)
    print("EXPECTED GESTURE CLASS FOLDERS (D01-D13):")
    print("-"*80)
    expected_classes = [f'D{i:02d}' for i in range(1, 14)]
    for class_name in expected_classes:
        class_path = ipn_path / class_name
        if class_path.exists() and class_path.is_dir():
            avi_count = len(list(class_path.glob("*.avi")))
            print(f"  ✓ {class_name}: {avi_count} AVI files")
        else:
            print(f"  ✗ {class_name}: NOT FOUND")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipn_dir', type=str, default='ipn_hand_dataset/frames',
                       help='Path to IPN dataset directory')
    args = parser.parse_args()
    
    check_structure(args.ipn_dir)