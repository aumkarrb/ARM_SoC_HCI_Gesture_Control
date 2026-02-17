#!/usr/bin/env python3
"""
IPN Hand Dataset Downloader
Downloads and extracts the IPN Hand gesture dataset
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"\n📦 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)
    print(f"✓ Extracted to {extract_to}")


def verify_dataset_structure(dataset_dir):
    """Verify the dataset has the expected structure"""
    expected_gestures = [
        "D01", "D02", "D03", "D04", "D05", "D06", "D07", 
        "D08", "D09", "D10", "D11", "D12", "D13"
    ]
    
    frames_dir = Path(dataset_dir) / "frames"
    if not frames_dir.exists():
        return False, "frames directory not found"
    
    found_gestures = [d.name for d in frames_dir.iterdir() if d.is_dir()]
    missing = set(expected_gestures) - set(found_gestures)
    
    if missing:
        return False, f"Missing gesture folders: {missing}"
    
    return True, "Dataset structure verified"


def main():
    """Main download function"""
    print("=" * 80)
    print("IPN HAND DATASET DOWNLOADER")
    print("=" * 80)
    
    # Create dataset directory
    dataset_dir = Path("ipn_hand_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    zip_path = dataset_dir / "Frames_resolution_640_480.zip"
    
    # Check if already downloaded
    if zip_path.exists():
        print(f"\n✓ Dataset zip file already exists: {zip_path}")
        print(f"  Size: {zip_path.stat().st_size / (1024**3):.2f} GB")
    else:
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print("\nThe IPN Hand dataset must be downloaded manually:")
        print("\n1. Visit: http://gibran.arandanos.mx/IPN_Hand_datasets.html")
        print("2. Download: 'Frames_resolution_640_480.zip' (~8 GB)")
        print(f"3. Place the zip file in: {dataset_dir.absolute()}")
        print("\nAfter downloading, run this script again to extract.")
        print("="*80)
        return
    
    # Check if already extracted
    frames_dir = dataset_dir / "frames"
    if frames_dir.exists():
        print(f"\n✓ Dataset already extracted: {frames_dir}")
        valid, msg = verify_dataset_structure(dataset_dir)
        if valid:
            print(f"✓ {msg}")
            print("\n" + "="*80)
            print("DATASET READY!")
            print("="*80)
            print(f"\nDataset location: {frames_dir.absolute()}")
            print("\nNext step: Run process_ipn_hand.py to extract keypoints")
            print("  python process_ipn_hand.py --ipn_dir ipn_hand_dataset/frames --output_dir ipn_processed")
        else:
            print(f"⚠ Warning: {msg}")
        return
    
    # Extract the dataset
    print("\n" + "="*80)
    print("EXTRACTING DATASET")
    print("="*80)
    try:
        extract_zip(zip_path, dataset_dir)
        
        # Verify extraction
        valid, msg = verify_dataset_structure(dataset_dir)
        if valid:
            print(f"\n✓ {msg}")
            print("\n" + "="*80)
            print("EXTRACTION COMPLETE!")
            print("="*80)
            print(f"\nDataset location: {frames_dir.absolute()}")
            print("\nDataset contains 13 gesture classes:")
            print("  D01: Pointing with one finger")
            print("  D02: Pointing with two fingers")
            print("  D03: Click with one finger")
            print("  D04: Click with two fingers")
            print("  D05: Throw up")
            print("  D06: Throw down")
            print("  D07: Throw left")
            print("  D08: Throw right")
            print("  D09: Open hand")
            print("  D10: Grab with hand")
            print("  D11: Expand with hand")
            print("  D12: Pinch with hand")
            print("  D13: Move with hand")
            print("\nNext step: Run process_ipn_hand.py to extract keypoints")
            print("  python process_ipn_hand.py --ipn_dir ipn_hand_dataset/frames --output_dir ipn_processed")
        else:
            print(f"\n⚠ Warning: {msg}")
            print("Please verify the dataset was extracted correctly.")
            
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        print("Please verify the zip file is not corrupted.")


if __name__ == "__main__":
    main()