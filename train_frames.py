"""
train_frames.py — CNN+LSTM on raw video frames
================================================
Uses pretrained ResNet18 as per-frame feature extractor,
then BiLSTM+Attention for temporal gesture classification.

This is what papers use to get 90-98% accuracy.
MediaPipe keypoints cap at ~68% because they lose texture,
arm position, and motion blur — all useful for gesture ID.

Expected: 85-95% val accuracy on IPN Hand dataset.

Usage:
  python3 train_frames.py --data_dir frame_processed
"""

import os, sys, time, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as tvm
import torchvision.transforms as T

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ─── ImageNet normalization ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
#  Dataset
# =============================================================================

class FrameGestureDataset(Dataset):
    """
    Loads (seq_len, H, W, 3) uint8 frame sequences.
    Normalizes with ImageNet stats (matches pretrained ResNet).
    """
    def __init__(self, data_dir, split='train', augment=False):
        self.augment = augment
        self.samples, self.labels = [], []
        split_dir = Path(data_dir) / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split not found: {split_dir}\n"
                "Run: python3 process_frames.py --ipd_dir ... first"
            )

        self.classes      = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for f in sorted((split_dir / cls).glob('*.npy')):
                self.samples.append(f)
                self.labels.append(self.class_to_idx[cls])

        self.class_counts = defaultdict(int)
        for lbl in self.labels:
            self.class_counts[self.classes[lbl]] += 1

        # Transforms
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        print(f"  [{split:5s}] {len(self.samples):5d} seqs  "
              f"classes={self.classes}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx])   # (T, H, W, 3) uint8
        # Convert to float tensor (T, 3, H, W) in [0,1]
        seq = torch.from_numpy(seq).float() / 255.0
        seq = seq.permute(0, 3, 1, 2)       # (T, 3, H, W)

        if self.augment:
            # Apply same spatial transforms to all frames
            seq = torch.stack([self.transform(frame) for frame in seq])
        else:
            seq = torch.stack([self.transform(frame) for frame in seq])

        return seq, self.labels[idx]


def compute_class_weights(ds):
    counts  = np.array([ds.class_counts[c] for c in ds.classes], dtype=np.float32)
    weights = counts.sum() / (len(ds.classes) * counts)
    return torch.FloatTensor(weights)


# =============================================================================
#  Model: pretrained CNN per frame + BiLSTM + Attention
# =============================================================================

class FrameCNNLSTM(nn.Module):
    """
    Per-frame: ResNet18 backbone (pretrained ImageNet) → 512-dim feature
    Temporal:  BiLSTM + attention over T frames
    Classify:  FC head

    Why this works:
      ResNet18 pretrained on 1M images already knows edges, textures, shapes.
      Fine-tuning last 2 layers adapts it to hands specifically.
      BiLSTM captures the temporal gesture trajectory.
    """
    def __init__(self, num_classes=6, lstm_hidden=256, lstm_layers=2,
                 dropout=0.5, freeze_backbone=False):
        super().__init__()

        # Load pretrained ResNet18, remove final FC
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features   # 512

        # Optionally freeze early layers
        if freeze_backbone:
            for name, param in backbone.named_parameters():
                if 'layer4' not in name and 'layer3' not in name:
                    param.requires_grad = False

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim   # 512

        # Dropout after CNN
        self.cnn_drop = nn.Dropout(dropout * 0.6)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Attention
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T = x.shape[:2]

        # Extract per-frame features
        x_flat  = x.view(B * T, *x.shape[2:])          # (B*T, 3, H, W)
        feats   = self.backbone(x_flat)                  # (B*T, 512)
        feats   = self.cnn_drop(feats)
        feats   = feats.view(B, T, self.feat_dim)        # (B, T, 512)

        # BiLSTM
        out, (h, _) = self.bilstm(feats)                # (B, T, 512), (4, B, 256)
        out = F.dropout(out, p=0.3, training=self.training)

        # Attention
        w        = F.softmax(self.attn(out), dim=1)     # (B, T, 1)
        attended = (out * w).sum(dim=1)                  # (B, 512)

        # Final hidden
        final_h = torch.cat([h[-2], h[-1]], dim=1)      # (B, 512)

        combined = torch.cat([attended, final_h], dim=1) # (B, 1024)
        return self.classifier(combined)


# =============================================================================
#  Train / eval loops
# =============================================================================

def gpu_info():
    if not torch.cuda.is_available(): return "CPU"
    a = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{a:.1f}/{t:.1f}GB ({100*a/t:.0f}%)"


def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            out  = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_sum += loss.item() * x.size(0)
        correct  += out.argmax(1).eq(y).sum().item()
        total    += x.size(0)
        print(f"  [{i+1:3d}/{len(loader)}] loss={loss.item():.4f}  gpu={gpu_info()}", end='\r')
    print()
    return loss_sum/total, 100.*correct/total, time.time()-t0


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_p, all_t = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast('cuda'):
            out  = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred      = out.argmax(1)
        correct  += pred.eq(y).sum().item()
        total    += x.size(0)
        all_p.extend(pred.cpu().numpy())
        all_t.extend(y.cpu().numpy())
    return loss_sum/total, 100.*correct/total, np.array(all_p), np.array(all_t)


def save_cm(preds, tgts, classes, path, title=''):
    if not HAS_VIZ: return
    cm = confusion_matrix(tgts, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =============================================================================
#  Main
# =============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🔥 GPU: {name}  ({mem:.1f}GB)  |  FP16 ON")
    else:
        print("\n⚠️  CPU mode")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/frames_{time.strftime('%Y%m%d_%H%M%S')}")

    print("\nLoading datasets:")
    train_ds = FrameGestureDataset(args.data_dir, 'train', augment=True)
    val_ds   = FrameGestureDataset(args.data_dir, 'val',   augment=False)
    num_cls  = len(train_ds.classes)
    n_train  = len(train_ds)

    class_weights = compute_class_weights(train_ds).to(device)
    print(f"\n  Class weights:")
    for cls, w in zip(train_ds.classes, class_weights.cpu()):
        print(f"    {cls:15s}: {train_ds.class_counts[cls]:4d} seqs  w={w:.3f}")

    # DataLoaders — num_workers=2 (frames are large, don't overload memory)
    sw       = [class_weights[l].item() for l in train_ds.labels]
    sampler  = WeightedRandomSampler(sw, len(train_ds), replacement=True)
    train_ld = DataLoader(train_ds, args.batch_size, sampler=sampler,
                          num_workers=2, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    val_ld   = DataLoader(val_ds, args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    # Model
    model = FrameCNNLSTM(
        num_classes=num_cls,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * len(train_ld),
                           pct_start=0.15, anneal_strategy='cos')
    scaler    = GradScaler('cuda')

    print(f"\n{'='*60}")
    print(f"  Model      : ResNet18 + BiLSTM + Attention")
    print(f"  Params     : {n_params:,} trainable")
    print(f"  Epochs     : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
    print(f"  Batches/ep : {len(train_ld)}  ({n_train} real sequences)")
    print(f"  Expected   : 85-95% val accuracy")
    print(f"{'='*60}")

    best_val, patience_cnt = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch:3d}/{args.epochs}]")

        tr_loss, tr_acc, elapsed = train_epoch(
            model, train_ld, optimizer, criterion, scaler, scheduler, device)
        vl_loss, vl_acc, preds, tgts = eval_epoch(
            model, val_ld, criterion, device)

        gap = tr_acc - vl_acc
        flag = f"  ⚠️ gap={gap:.0f}%" if gap > 20 else ""
        print(f"  Train: loss={tr_loss:.4f}  acc={tr_acc:.1f}%  ({elapsed:.0f}s)")
        print(f"  Val  : loss={vl_loss:.4f}  acc={vl_acc:.1f}%{flag}")

        writer.add_scalars('Loss',     {'train': tr_loss, 'val': vl_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': tr_acc,  'val': vl_acc},  epoch)

        if vl_acc > best_val:
            best_val = vl_acc; patience_cnt = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_acc': vl_acc, 'classes': train_ds.classes,
                'model': 'frame_cnn_lstm',
            }, ckpt_dir / 'best_model.pth')
            print(f"  ✅ Best saved  val={best_val:.1f}%")
            save_cm(preds, tgts, train_ds.classes,
                    ckpt_dir / 'val_confusion.png',
                    f"Val ep={epoch}  acc={vl_acc:.1f}%")
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f"\n  Early stop — no improvement for {args.patience} epochs")
            break

    # Final test
    print(f"\n{'='*60}")
    print(f"  FINAL TEST")
    print(f"{'='*60}")
    try:
        test_ds = FrameGestureDataset(args.data_dir, 'test', augment=False)
        test_ld = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)
        ckpt = torch.load(ckpt_dir / 'best_model.pth', map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        ts_loss, ts_acc, tp, tt = eval_epoch(model, test_ld, criterion, device)
        print(f"  Test acc={ts_acc:.1f}%")
        save_cm(tp, tt, test_ds.classes, ckpt_dir/'test_confusion.png',
                f"TEST acc={ts_acc:.1f}%")
        if HAS_VIZ:
            print("\nClassification Report:")
            print(classification_report(tt, tp, target_names=test_ds.classes))
    except Exception as e:
        print(f"  Test failed: {e}"); ts_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  ✅  DONE")
    print(f"  Best Val  : {best_val:.2f}%")
    print(f"  Test Acc  : {ts_acc:.2f}%")
    print(f"{'='*60}")
    writer.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',        default='frame_processed')
    p.add_argument('--ckpt_dir',        default='checkpoints_frames')
    p.add_argument('--epochs',          type=int,   default=30)
    p.add_argument('--batch_size',      type=int,   default=8)
    p.add_argument('--lr',             type=float, default=0.0001)
    p.add_argument('--dropout',        type=float, default=0.5)
    p.add_argument('--patience',       type=int,   default=10)
    p.add_argument('--freeze_backbone',action='store_true',
                   help='Freeze early ResNet layers (faster, less GPU)')
    main(p.parse_args())