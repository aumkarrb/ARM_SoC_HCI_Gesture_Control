import torch

from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from utils.heatmap_decode import heatmaps_to_keypoints
from filters.one_euro import OneEuroFilter
from geometry.hand_geometry import extract_geometry_features
from fsm.gesture_fsm import GestureFSM


device = "cuda" if torch.cuda.is_available() else "cpu"

model = MobileNetV2Heatmap(num_keypoints=21).to(device)
model.eval()

filter = OneEuroFilter(freq=30, beta=0.4)
fsm = GestureFSM(stable_frames=5)

x = torch.randn(1, 1, 224, 224)

with torch.no_grad():
    for frame in range(20):
        heatmaps = model(x.to(device))
        keypoints = heatmaps_to_keypoints(heatmaps)[0]
        keypoints = filter.filter(keypoints)

        features = extract_geometry_features(keypoints)
        gesture = fsm.update(features)

        print(f"Frame {frame} → Gesture: {gesture}")
