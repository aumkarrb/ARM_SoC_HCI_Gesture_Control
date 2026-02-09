"""
Direct Gesture Classification on HaGRID Images
Train a CNN directly on the images (no keypoints needed)
This will work MUCH better than the keypoint approach!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
HAGRID_ROOT = Path(r"C:\Users\Soham\Downloads\Datasets\hagrid-sample-30k-384p\dataset\train")
OUTPUT_MODEL = r"C:\Users\Soham\Downloads\handpose_project\outputs\gesture_cnn.pth"

GESTURES = ["two_up", "two_up_inverted", "fist", "ok", "palm", "stop"]
NUM_CLASSES = len(GESTURES)

BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class GestureDataset(Dataset):
    def __init__(self, root_dir, gestures, transform=None):
        self.data = []
        self.transform = transform
        self.gesture_to_idx = {g: i for i, g in enumerate(gestures)}
        
        # Load all image paths
        for gesture in gestures:
            gesture_path = root_dir / gesture
            if gesture_path.exists():
                images = list(gesture_path.glob("*.jpg")) + list(gesture_path.glob("*.png"))
                for img_path in images:
                    self.data.append((str(img_path), self.gesture_to_idx[gesture]))
        
        print(f"Loaded {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading dataset...")
full_dataset = GestureDataset(HAGRID_ROOT, GESTURES, transform=train_transform)

# Split train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

from torch.utils.data import random_split
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Update val dataset transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {train_size}")
print(f"Val samples: {val_size}")

# Count per class
print("\nSamples per gesture:")
for gesture in GESTURES:
    count = sum(1 for _, label in full_dataset.data if label == full_dataset.gesture_to_idx[gesture])
    print(f"  {gesture:20s}: {count}")

# --------------------------------------------------
# MODEL (MobileNetV2 - Same as before but for classification)
# --------------------------------------------------
print("\nInitializing model...")
model = models.mobilenet_v2(pretrained=True)

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, NUM_CLASSES)
)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

print("✓ Model ready")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING GESTURE CLASSIFIER (Direct CNN)")
print("=" * 60)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
    
    train_acc = 100.0 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.0 * val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_acc)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'gesture_to_idx': full_dataset.gesture_to_idx,
            'val_acc': val_acc,
            'epoch': epoch
        }, OUTPUT_MODEL)
        print(f"  ✓ Best model saved! Val Acc: {val_acc:.2f}%")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Model saved to: {OUTPUT_MODEL}")
print("=" * 60)
print("\nNext step: Run inference_gestures_cnn.py for real-time recognition!")
print("=" * 60)