import torch

from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from utils.heatmap_decode import heatmaps_to_keypoints
from filters.one_euro import OneEuroFilter


# -----------------------------
# Model
# -----------------------------
model = MobileNetV2Heatmap(num_keypoints=21)
model.eval()


# -----------------------------
# 🔹 One-Euro Filter (INITIALISE ONCE)
# -----------------------------
keypoint_filter = OneEuroFilter(
    freq=30,          # assume ~30 FPS
    min_cutoff=1.0,   # smoothing when still
    beta=0.4,         # responsiveness to motion
    d_cutoff=1.0
)


# -----------------------------
# Test inference
# -----------------------------
x = torch.randn(1, 1, 224, 224)

with torch.no_grad():
    heatmaps = model(x)

    # Heatmaps → keypoints
    keypoints = heatmaps_to_keypoints(heatmaps)

    # 🔹 Apply One-Euro filter HERE
    keypoints_smoothed = keypoint_filter.filter(keypoints[0])

print("Keypoints shape:", keypoints_smoothed.shape)
print("First keypoint:", keypoints_smoothed[0])
