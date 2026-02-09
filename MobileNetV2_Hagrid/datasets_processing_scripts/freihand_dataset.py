import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils.heatmaps import generate_heatmaps


class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, img_size=224, heatmap_size=7):
        self.root = Path(root_dir)
        self.img_size = img_size
        self.heatmap_size = heatmap_size

        self.rgb_dir = self.root / "training" / "rgb"
        self.xyz_path = self.root / "training_xyz.json"
        self.K_path = self.root / "training_K.json"

        assert self.rgb_dir.exists(), "training/rgb not found"
        assert self.xyz_path.exists(), "training_xyz.json not found"
        assert self.K_path.exists(), "training_K.json not found"

        with open(self.xyz_path) as f:
            self.xyz = json.load(f)

        with open(self.K_path) as f:
            self.K = json.load(f)

        # THIS is the authoritative dataset length
        self.length = len(self.xyz)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # -------- Load image by index --------
        img_path = self.rgb_dir / f"{idx:08d}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        # -------- 3D keypoints --------
        xyz = np.array(self.xyz[idx])  # [21, 3]

        # -------- Camera intrinsics --------
        K = np.array(self.K[idx])  # [3, 3]

        # -------- Project to 2D --------
        uv = xyz @ K.T
        uv = uv[:, :2] / uv[:, 2:]

        uv = uv / self.img_size
        uv = torch.from_numpy(uv).float()

        # -------- Heatmaps --------
        heatmaps = generate_heatmaps(
            uv,
            out_size=(self.heatmap_size, self.heatmap_size),
            sigma=1.0
        )

        return img, heatmaps
