import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.hagrid_clean_dataset import HaGRIDCleanDataset
from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MobileNetV2Heatmap(num_keypoints=21).to(device)

dataset = HaGRIDCleanDataset(HAGRID_ROOT, split="train")
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for step, (img, target) in enumerate(loader):
    img = img.to(device)
    target = target.to(device)

    pred = model(img)
    loss = F.mse_loss(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")

    if step == 10:
        break
