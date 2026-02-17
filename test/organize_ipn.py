import os
from pathlib import Path
import shutil

# CHANGE THIS

base_dir = Path(r"C:\Users\Soham\Downloads\IPD dataset\videos\videos")

for video in base_dir.glob("*.avi"):
    name = video.stem  # remove .avi
    
    parts = name.split("_")
    if len(parts) < 2:
        continue
    
    gesture_number = parts[1]  # second part
    
    try:
        gesture_int = int(gesture_number)
    except:
        continue
    
    # Keep only D01–D13
    if 1 <= gesture_int <= 13:
        folder_name = f"D{gesture_int:02d}"
        class_folder = base_dir / folder_name
        class_folder.mkdir(exist_ok=True)
        
        shutil.move(str(video), str(class_folder / video.name))

print("Organization complete.")
