from pathlib import Path

# CHANGE THIS ONLY IF DATASET MOVES
HAGRID_ROOT = Path(
    r"C:\Users\Soham\Downloads\Datasets\hagrid-sample-30k-384p\dataset"
)

# Model settings
IMG_SIZE = 224
NUM_CLASSES = 6       # for now (classification baseline)
NUM_KEYPOINTS = 21    # future heatmap model

# Training (RTX 2050 safe)
BATCH_SIZE = 8
NUM_WORKERS = 2
