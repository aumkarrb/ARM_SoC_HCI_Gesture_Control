import torch
from models.mobilenetv2_heatmap import MobileNetV2Heatmap

model = MobileNetV2Heatmap(num_keypoints=21)
model.eval()

x = torch.randn(1, 1, 224, 224)
y = model(x)

print("Output heatmaps:", y.shape)
