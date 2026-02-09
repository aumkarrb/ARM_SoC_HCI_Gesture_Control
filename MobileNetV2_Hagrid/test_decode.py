import torch
from utils.heatmap_decode import heatmaps_to_keypoints

heatmaps = torch.zeros((1, 21, 7, 7))
heatmaps[0, 0, 3, 4] = 1.0  # fake peak

keypoints = heatmaps_to_keypoints(heatmaps)
print(keypoints[0, 0])
