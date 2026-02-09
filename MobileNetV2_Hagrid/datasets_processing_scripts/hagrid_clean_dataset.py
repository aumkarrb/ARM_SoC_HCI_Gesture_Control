import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.fake_keypoints import generate_fake_hand_keypoints
from utils.heatmaps import generate_heatmaps

class HaGRIDCleanDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=224):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.samples = []

        split_dir = self.root_dir / split
        assert split_dir.exists(), f"{split_dir} does not exist"

        self.classes = sorted(
            [d.name for d in split_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for img_path in (split_dir / cls).glob("*.jpg"):
                self.samples.append((img_path, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls = self.samples[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        # Fake keypoints → heatmaps
        keypoints = generate_fake_hand_keypoints()
        heatmaps = generate_heatmaps(keypoints)

        return img, heatmaps