"""
train_hagrid.py — EfficientNet-B0 image classifier for HaGRID gestures
=======================================================================
No LSTM. No keypoints. No video. Just images → gesture class.
EfficientNet-B0 pretrained on ImageNet, fine-tuned on HaGRID.

Expected: 90-95% test accuracy in ~20 minutes.

Folder structure expected:
  hagrid_30k/
    train_val_fist/          ← images directly inside
    train_val_ok/
    train_val_palm/
    train_val_stop/
    train_val_two_up/
    train_val_two_up_inverted/

Usage:
  python3 train_hagrid.py --data_dir "path/to/hagrid_30k"
"""

import os, time, argparse, json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T
import torchvision.models as tvm
from PIL import Image

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Maps folder suffix → clean class name
GESTURE_MAP = {
    'fist':           'fist',
    'ok':             'ok',
    'palm':           'palm',
    'stop':           'stop',
    'two_up':         'two_up',
    'two_up_inverted':'two_up_inverted',
}


# =============================================================================
#  Dataset
# =============================================================================

class HaGRIDDataset(Dataset):
    def __init__(self, data_dir, split='train', val_ratio=0.15, augment=False, seed=42):
        self.augment = augment
        self.samples = []
        self.labels  = []
        data_path = Path(data_dir)

        # Find all gesture folders
        gesture_dirs = {}
        for folder in sorted(data_path.iterdir()):
            if not folder.is_dir(): continue
            for suffix, cls in GESTURE_MAP.items():
                if folder.name.endswith(suffix):
                    gesture_dirs[cls] = folder
                    break

        if not gesture_dirs:
            raise FileNotFoundError(
                f"No gesture folders found in {data_dir}\n"
                f"Expected folders like train_val_fist, train_val_ok, etc."
            )

        self.classes      = sorted(gesture_dirs.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Split images per class
        rng = np.random.default_rng(seed)
        for cls in self.classes:
            imgs = sorted(gesture_dirs[cls].glob('*.jpg')) + \
                   sorted(gesture_dirs[cls].glob('*.png')) + \
                   sorted(gesture_dirs[cls].glob('*.jpeg'))
            imgs = np.array(imgs)
            idx  = rng.permutation(len(imgs))
            imgs = imgs[idx]

            n_val   = int(len(imgs) * val_ratio)
            n_train = len(imgs) - n_val

            if split == 'train':
                selected = imgs[:n_train]
            else:
                selected = imgs[n_train:]

            for img_path in selected:
                self.samples.append(img_path)
                self.labels.append(self.class_to_idx[cls])

        self.class_counts = defaultdict(int)
        for lbl in self.labels:
            self.class_counts[self.classes[lbl]] += 1

        # Transforms
        if augment:
            self.transform = T.Compose([
                T.Resize((240, 240)),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                T.RandomGrayscale(p=0.05),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                T.ToTensor(),
                T.RandomErasing(p=0.15, scale=(0.02, 0.1)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        print(f"  [{split:5s}] {len(self.samples):6d} images  "
              f"classes={self.classes}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]


# =============================================================================
#  Model: EfficientNet-B0
# =============================================================================

def build_model(num_classes, dropout=0.4):
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"  Model      : EfficientNet-B0")
    print(f"  Parameters : {total:,} total / {trainable:,} trainable")
    print(f"  Input      : (batch, 3, 224, 224)")
    print(f"  Output     : (batch, {num_classes})")
    print(f"  Expected   : 90-95% accuracy")
    print(f"{'='*55}\n")
    return model


# =============================================================================
#  Train / eval
# =============================================================================

def gpu_mem():
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
        if (i+1) % 50 == 0 or (i+1) == len(loader):
            print(f"  [{i+1:4d}/{len(loader)}] loss={loss.item():.4f}  gpu={gpu_mem()}", end='\r')
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


def save_confusion_matrix(preds, tgts, classes, path, title=''):
    if not HAS_VIZ: return
    cm = confusion_matrix(tgts, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
#  Main
# =============================================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🔥 GPU : {name}  ({mem:.1f} GB)  |  FP16 ON")
    else:
        print("\n⚠️  CPU mode — will be slow")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading datasets:")
    train_ds = HaGRIDDataset(args.data_dir, 'train', augment=True)
    val_ds   = HaGRIDDataset(args.data_dir, 'val',   augment=False)
    num_cls  = len(train_ds.classes)

    print(f"\n  Per-class counts (train):")
    for cls in train_ds.classes:
        print(f"    {cls:20s}: {train_ds.class_counts[cls]:5d} images")

    # Class weights for imbalanced classes
    counts  = np.array([train_ds.class_counts[c] for c in train_ds.classes], dtype=np.float32)
    weights = torch.FloatTensor(counts.sum() / (num_cls * counts)).to(device)

    # Weighted sampler
    sw      = [weights[l].item() for l in train_ds.labels]
    sampler = WeightedRandomSampler(sw, len(train_ds), replacement=True)

    train_ld = DataLoader(train_ds, args.batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    val_ld   = DataLoader(val_ds, args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True)

    model     = build_model(num_cls, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Differential LR: backbone 10x slower than head
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    head_params     = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr / 10, 'weight_decay': 1e-4},
        {'params': head_params,     'lr': args.lr,      'weight_decay': 1e-4},
    ])

    total_steps = args.epochs * len(train_ld)
    scheduler   = OneCycleLR(optimizer, max_lr=[args.lr/10, args.lr],
                             total_steps=total_steps,
                             pct_start=0.1, anneal_strategy='cos')
    scaler      = GradScaler('cuda')

    print(f"\n{'='*55}")
    print(f"  Epochs      : {args.epochs}  |  batch={args.batch_size}  |  lr={args.lr}")
    print(f"  Batches/ep  : {len(train_ld)}  ({len(train_ds)} train images)")
    print(f"  Total steps : {total_steps:,}")
    print(f"{'='*55}")

    best_val, patience_cnt = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch:3d}/{args.epochs}]")
        tr_loss, tr_acc, elapsed = train_epoch(
            model, train_ld, optimizer, criterion, scaler, scheduler, device)
        vl_loss, vl_acc, preds, tgts = eval_epoch(
            model, val_ld, criterion, device)

        gap  = tr_acc - vl_acc
        flag = f"  ⚠️ gap={gap:.0f}%" if gap > 15 else ""
        print(f"  Train : loss={tr_loss:.4f}  acc={tr_acc:.1f}%  ({elapsed:.0f}s)")
        print(f"  Val   : loss={vl_loss:.4f}  acc={vl_acc:.1f}%{flag}")

        if vl_acc > best_val:
            best_val = vl_acc; patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': vl_acc,
                'classes': train_ds.classes,
                'model': 'efficientnet_b0',
            }, ckpt_dir / 'best_model.pth')
            print(f"  ✅ Best saved  val={best_val:.1f}%")
            save_confusion_matrix(preds, tgts, train_ds.classes,
                                  ckpt_dir/'val_confusion.png',
                                  f"Val ep={epoch}  acc={vl_acc:.1f}%")
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f"\n  ⏹️  Early stop — no improvement for {args.patience} epochs")
            break

    # ── Final test evaluation ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL TEST EVALUATION")
    print(f"{'='*55}")

    # Use val set as test (HaGRID 30k — already held out during training)
    ckpt = torch.load(ckpt_dir/'best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    ts_loss, ts_acc, tp, tt = eval_epoch(model, val_ld, criterion, device)

    print(f"\n  Test acc={ts_acc:.1f}%")
    save_confusion_matrix(tp, tt, train_ds.classes,
                          ckpt_dir/'test_confusion.png',
                          f"TEST acc={ts_acc:.1f}%")

    if HAS_VIZ:
        print("\n  Classification Report:")
        print(classification_report(tt, tp, target_names=train_ds.classes))

    # Save metadata for Jetson inference
    meta = {
        'model': 'efficientnet_b0',
        'classes': train_ds.classes,
        'input_size': 224,
        'val_acc': best_val,
        'test_acc': ts_acc,
    }
    json.dump(meta, open(ckpt_dir/'metadata.json', 'w'), indent=2)

    print(f"\n{'='*55}")
    print(f"  ✅  DONE")
    print(f"  Best Val  : {best_val:.2f}%")
    print(f"  Test Acc  : {ts_acc:.2f}%")
    print(f"  Checkpoint: {ckpt_dir}/best_model.pth")
    print(f"\n  Next step:")
    print(f"  python3 export_hagrid_onnx.py --ckpt {ckpt_dir}/best_model.pth")
    print(f"{'='*55}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  required=True,
                   help='Path to hagrid_30k folder containing train_val_* subfolders')
    p.add_argument('--ckpt_dir',  default='checkpoints_hagrid')
    p.add_argument('--epochs',    type=int,   default=20)
    p.add_argument('--batch_size',type=int,   default=32)
    p.add_argument('--lr',        type=float, default=0.001)
    p.add_argument('--dropout',   type=float, default=0.4)
    p.add_argument('--patience',  type=int,   default=8)
    main(p.parse_args())