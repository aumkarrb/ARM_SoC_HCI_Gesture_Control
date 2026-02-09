import torch
from geometry.hand_geometry import extract_geometry_features

# Fake keypoints for test
kp = torch.rand(21, 2)

features = extract_geometry_features(kp)

for k, v in features.items():
    print(k, v)
