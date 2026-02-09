"""
Extract Hand Keypoints from HaGRID Dataset - CORRECTED STRUCTURE
Uses your trained FreiHAND model to extract 21 keypoints from all HaGRID images
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, r"C:\Users\Soham\Downloads\handpose_project")
from models.mobilenetv2_heatmap import MobileNetV2Heatmap

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
# CORRECTED PATH: Images are in dataset/train/ not dataset/
HAGRID_ROOT = Path(r"C:\Users\Soham\Downloads\Datasets\hagrid-sample-30k-384p\dataset\train")
MODEL_PATH = r"C:\Users\Soham\Downloads\handpose_project\outputs\best_freihand_model.pth"
OUTPUT_PATH = r"C:\Users\Soham\Downloads\handpose_project\hagrid_keypoints.json"

IMG_SIZE = 224
HEATMAP_SIZE = 7

GESTURES = ["two_up", "two_up_inverted", "fist", "ok", "palm", "stop"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# --------------------------------------------------
# VERIFY PATHS
# --------------------------------------------------
print("Verifying HaGRID dataset location...")
print(f"Looking in: {HAGRID_ROOT}")

if not HAGRID_ROOT.exists():
    print(f"❌ ERROR: HaGRID train folder not found: {HAGRID_ROOT}")
    sys.exit(1)

print("✓ HaGRID train folder found")

# Check gesture folders
print("\nChecking gesture folders:")
found_gestures = []
for gesture in GESTURES:
    gesture_path = HAGRID_ROOT / gesture
    if gesture_path.exists():
        num_images = len(list(gesture_path.glob("*.jpg"))) + len(list(gesture_path.glob("*.png")))
        print(f"  ✓ {gesture:20s}: {num_images} images")
        found_gestures.append(gesture)
    else:
        print(f"  ✗ {gesture:20s}: FOLDER NOT FOUND")

if not found_gestures:
    print("\n❌ ERROR: No gesture folders found!")
    print(f"\nContents of {HAGRID_ROOT}:")
    for item in HAGRID_ROOT.iterdir():
        print(f"  {item.name}")
    sys.exit(1)

print(f"\n✓ Found {len(found_gestures)} gesture folders")

# --------------------------------------------------
# LOAD FREIHAND MODEL
# --------------------------------------------------
print("\nLoading FreiHAND keypoint model...")
model = MobileNetV2Heatmap(num_keypoints=21, in_channels=1)
checkpoint = torch.load(MODEL_PATH, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")

model.to(device)
model.eval()
print("✓ Model ready\n")

# --------------------------------------------------
# KEYPOINT EXTRACTION
# --------------------------------------------------
def extract_keypoints(image_path):
    """Extract 21 keypoints from an image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            heatmaps = model(img_tensor)
        
        # Extract keypoints
        heatmaps = heatmaps.squeeze(0).cpu().numpy()
        keypoints = []
        
        for i in range(21):
            hm = heatmaps[i]
            max_val = hm.max()
            
            if max_val < 0.1:  # Low confidence threshold
                keypoints.extend([0.0, 0.0])  # Placeholder
            else:
                y, x = np.unravel_index(hm.argmax(), hm.shape)
                # Normalize to 0-1 range
                x_norm = x / HEATMAP_SIZE
                y_norm = y / HEATMAP_SIZE
                keypoints.extend([x_norm, y_norm])
        
        return keypoints  # 42 values (21 keypoints × 2 coordinates)
    
    except Exception as e:
        return None

# --------------------------------------------------
# PROCESS ALL IMAGES
# --------------------------------------------------
print("=" * 60)
print("Extracting keypoints from HaGRID images...")
print("=" * 60)

dataset = []
total_processed = 0
total_failed = 0

for gesture in found_gestures:
    gesture_path = HAGRID_ROOT / gesture
    
    # Get all images in gesture folder
    image_files = list(gesture_path.glob("*.jpg")) + list(gesture_path.glob("*.png"))
    
    print(f"\n📁 Processing '{gesture}': {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc=f"  {gesture}"):
        keypoints = extract_keypoints(img_path)
        
        if keypoints is not None:
            dataset.append({
                "gesture": gesture,
                "keypoints": keypoints,
                "image_path": str(img_path)
            })
            total_processed += 1
        else:
            total_failed += 1

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"Total images processed: {total_processed}")
print(f"Failed: {total_failed}")

if total_processed > 0:
    success_rate = total_processed/(total_processed+total_failed)*100
    print(f"Success rate: {success_rate:.1f}%")
    
    # Count per gesture
    print("\nImages per gesture:")
    for gesture in found_gestures:
        count = sum(1 for item in dataset if item['gesture'] == gesture)
        print(f"  {gesture:20s}: {count:5d} images")
    
    # Save to JSON
    print(f"\nSaving keypoints to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(dataset, f)
    
    print("✓ Keypoints saved successfully!")
    print("=" * 60)
    print("\n🎯 Next step: Run train_gesture_classifier.py")
    print("=" * 60)
else:
    print("\n❌ No images were processed successfully!")