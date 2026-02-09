"""
Train Gesture Classifier on HaGRID Keypoints
Small neural network: 42 keypoint coordinates → 6 gesture classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
KEYPOINTS_PATH = r"C:\Users\Soham\Downloads\handpose_project\hagrid_keypoints.json"
OUTPUT_MODEL = r"C:\Users\Soham\Downloads\handpose_project\outputs\gesture_classifier.pth"

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

GESTURES = ["two_up", "two_up_inverted", "fist", "ok", "palm", "stop"]
NUM_CLASSES = len(GESTURES)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# --------------------------------------------------
# DATASET
# --------------------------------------------------
class KeypointDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.gesture_to_idx = {g: i for i, g in enumerate(GESTURES)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        keypoints = torch.FloatTensor(item['keypoints'])
        label = self.gesture_to_idx[item['gesture']]
        return keypoints, label

# --------------------------------------------------
# MODEL
# --------------------------------------------------
class GestureClassifier(nn.Module):
    def __init__(self, input_size=42, num_classes=6):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading keypoints...")
with open(KEYPOINTS_PATH, 'r') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} samples")

# Count per class
print("\nSamples per gesture:")
for gesture in GESTURES:
    count = sum(1 for item in data if item['gesture'] == gesture)
    print(f"  {gesture:20s}: {count}")

# Create dataset
dataset = KeypointDataset(data)

# Split train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain samples: {train_size}")
print(f"Val samples: {val_size}")

# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------
print("\nInitializing model...")
model = GestureClassifier(input_size=42, num_classes=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

print("✓ Model ready")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING GESTURE CLASSIFIER")
print("=" * 60)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for keypoints, labels in train_loader:
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100.0 * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.0 * val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_loss)
    
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
            'gesture_to_idx': dataset.gesture_to_idx,
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
print("\nNext step: Run inference_gestures_final.py for real-time recognition!")
print("=" * 60)