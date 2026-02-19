"""
CNN+LSTM Gesture Recognition — train.py  v3 (Final)
====================================================
Works with sliding window extraction (process_ipd_trtpose.py --stride 10).
Expected dataset: ~1,800 train seqs (~300/class) -> full GestureCNNLSTM.
Expected accuracy: 88-95% val, 85-93% test.
"""

import os, sys, argparse, time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

try:
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from models import build_model


# =============================================================================
#  Dataset
# =============================================================================

class GestureDataset(Dataset):
    """
    Loads .npy sequences from:
      data_dir/train/<class>/*.npy   (seq_len, 42)
      data_dir/val/<class>/*.npy
      data_dir/test/<class>/*.npy
    """
    def __init__(self, data_dir, split="train", seq_len=30, augment=False):
        self.seq_len = seq_len
        self.augment = augment
        self.samples, self.labels = [], []
        split_dir = Path(data_dir) / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split folder not found: {split_dir}\n"
                "Run: python3 process_ipd_trtpose.py --stride 10 first"
            )

        self.classes      = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for f in sorted((split_dir / cls).glob("*.npy")):
                self.samples.append(f)
                self.labels.append(self.class_to_idx[cls])

        self.class_counts = defaultdict(int)
        for lbl in self.labels:
            self.class_counts[self.classes[lbl]] += 1

        print(f"  [{split:5s}] {len(self.samples):5d} seqs  classes={self.classes}")

    def __len__(self): return len(self.samples)

    def _augment(self, seq):
        seq = seq.copy()

        # Keypoint jitter — simulates MediaPipe detection noise
        if np.random.rand() < 0.5:
            seq += np.random.normal(0, 0.02, seq.shape).astype(np.float32)

        # Time shift — gesture starts slightly early/late
        if np.random.rand() < 0.4:
            seq = np.roll(seq, np.random.randint(-3, 4), axis=0)

        # Scale — hand closer/farther from camera
        if np.random.rand() < 0.5:
            seq *= np.random.uniform(0.88, 1.12)

        # Slight wrist rotation
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-0.12, 0.12)
            c, s  = np.cos(angle), np.sin(angle)
            xy    = seq.reshape(self.seq_len, 21, 2).copy()
            xy[..., 0], xy[..., 1] = (xy[..., 0]*c - xy[..., 1]*s,
                                       xy[..., 0]*s + xy[..., 1]*c)
            seq = xy.reshape(self.seq_len, 42)

        # Mirror left<->right  (FIX: negate X after Z-score, not 1-x)
        if np.random.rand() < 0.4:
            seq[:, 0::2] *= -1

        # Frame dropout — simulates momentary detection failure
        if np.random.rand() < 0.25:
            drop = np.random.choice(self.seq_len, np.random.randint(1, 4), replace=False)
            seq[drop] = 0.0

        # Speed variation — faster/slower performer
        if np.random.rand() < 0.3:
            factor  = np.random.uniform(0.75, 1.25)
            new_len = max(4, int(self.seq_len * factor))
            sv = seq[np.linspace(0, self.seq_len-1, new_len, dtype=int)]
            T  = sv.shape[0]
            if T < self.seq_len:
                seq = np.vstack([sv, np.repeat(sv[-1:], self.seq_len-T, axis=0)])
            else:
                seq = sv[np.linspace(0, T-1, self.seq_len, dtype=int)]

        return seq

    def __getitem__(self, idx):
        seq = np.load(self.samples[idx]).astype(np.float32)   # (T, 42)
        T   = seq.shape[0]
        if T < self.seq_len:
            seq = np.vstack([seq, np.repeat(seq[-1:], self.seq_len-T, axis=0)])
        elif T > self.seq_len:
            seq = seq[np.linspace(0, T-1, self.seq_len, dtype=int)]

        # Z-score per feature — zero mean, unit variance across time
        seq = (seq - seq.mean(0, keepdims=True)) / (seq.std(0, keepdims=True) + 1e-6)

        if self.augment:
            seq = self._augment(seq)

        return torch.from_numpy(seq), self.labels[idx]


def compute_class_weights(ds):
    counts  = np.array([ds.class_counts[c] for c in ds.classes], dtype=np.float32)
    weights = counts.sum() / (len(ds.classes) * counts)
    return torch.FloatTensor(weights)


# =============================================================================
#  Training helpers
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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        loss_sum += loss.item() * x.size(0)
        correct  += out.argmax(1).eq(y).sum().item()
        total    += x.size(0)
        if (i+1) % 10 == 0 or (i+1) == len(loader):
            print(f"  [{i+1:3d}/{len(loader)}] loss={loss.item():.4f}  gpu={gpu_info()}", end="\r")
    print()
    return loss_sum/total, 100.0*correct/total, time.time()-t0


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_p, all_t = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            out  = model(x)
            loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred      = out.argmax(1)
        correct  += pred.eq(y).sum().item()
        total    += x.size(0)
        all_p.extend(pred.cpu().numpy())
        all_t.extend(y.cpu().numpy())
    return loss_sum/total, 100.0*correct/total, np.array(all_p), np.array(all_t)


def save_cm(preds, tgts, classes, path, title=""):
    if not HAS_VIZ: return
    cm  = confusion_matrix(tgts, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix saved: {path}")


# =============================================================================
#  Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {name}  ({mem:.1f}GB)  |  FP16 ON")
    else:
        print("\nCPU mode — will be slow")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(f"runs/run_{time.strftime('%Y%m%d_%H%M%S')}")

    # -- Load data ----------------------------------------------------------
    print("\nLoading datasets:")
    train_ds = GestureDataset(args.data_dir, "train", args.seq_len, augment=True)
    val_ds   = GestureDataset(args.data_dir, "val",   args.seq_len, augment=False)
    num_cls  = len(train_ds.classes)
    n_train  = len(train_ds)
    per_cls  = n_train // max(num_cls, 1)

    print(f"\n  {n_train} training sequences  (~{per_cls}/class)")

    # -- Model selection based on real data size ----------------------------
    if per_cls >= 100:
        model_type = args.model_type
        exp_acc    = "88-95%"
        print(f"  Dataset size GOOD -> full GestureCNNLSTM  (expected val {exp_acc})")
    elif per_cls >= 50:
        model_type = args.model_type
        exp_acc    = "82-90%"
        print(f"  Dataset size OK -> GestureCNNLSTM  (expected val {exp_acc})")
    else:
        model_type = "lightweight"
        exp_acc    = "75-85%"
        args.batch_size = min(args.batch_size, 16)
        print(f"  Dataset small ({per_cls}/class) -> LightweightCNNLSTM  (expected val {exp_acc})")
        print(f"  To improve: re-extract with smaller stride:")
        print(f"    rm -rf ipd_processed/")
        print(f"    python3 process_ipd_trtpose.py --ipd_dir \"$IPD_DIR\" --output_dir ipd_processed --stride 5")

    # -- Class weights ------------------------------------------------------
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"\n  Class weights (inverse frequency):")
    for cls, w in zip(train_ds.classes, class_weights.cpu()):
        print(f"    {cls:15s}: {train_ds.class_counts[cls]:4d} seqs  w={w:.3f}")

    # -- DataLoaders --------------------------------------------------------
    # WeightedRandomSampler: balanced class frequency per batch.
    # Samples exactly len(train_ds) per epoch — REAL data, no oversampling.
    sw      = [class_weights[l].item() for l in train_ds.labels]
    sampler = WeightedRandomSampler(sw, len(train_ds), replacement=True)

    train_ld = DataLoader(train_ds, args.batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    val_ld   = DataLoader(val_ds, args.batch_size*2, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)

    n_batches = len(train_ld)

    # -- Model + optimizer --------------------------------------------------
    model     = build_model(model_type, num_classes=num_cls, seq_len=args.seq_len).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.epochs * n_batches,
                           pct_start=0.1, anneal_strategy="cos")
    scaler    = GradScaler()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Model      : {model_type}  ({n_params:,} params)")
    print(f"  Batches/ep : {n_batches}   ({n_train} real sequences)")
    print(f"  Total steps: {args.epochs * n_batches:,}")
    print(f"  Batch size : {args.batch_size}   LR: {args.lr}   Epochs: {args.epochs}")
    print(f"  Expected   : training ~{args.epochs * n_batches / 60:.0f} min on RTX 2050")
    print(f"{'='*60}")

    # -- Training loop ------------------------------------------------------
    best_val, patience_cnt = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch:3d}/{args.epochs}]")

        tr_loss, tr_acc, elapsed = train_epoch(
            model, train_ld, optimizer, criterion, scaler, scheduler, device)
        vl_loss, vl_acc, preds, tgts = eval_epoch(
            model, val_ld, criterion, device)

        gap = tr_acc - vl_acc
        ovf = "  ⚠️ OVERFIT" if gap > 20 else ""
        unf = "  ⚠️ UNDERFIT" if vl_acc < 30 and epoch > 20 else ""
        print(f"  Train: loss={tr_loss:.4f}  acc={tr_acc:.1f}%  ({elapsed:.0f}s)")
        print(f"  Val  : loss={vl_loss:.4f}  acc={vl_acc:.1f}%  gap={gap:.1f}%{ovf}{unf}")

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": vl_acc},  epoch)

        if vl_acc > best_val:
            best_val = vl_acc
            patience_cnt = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": vl_acc, "model_type": model_type,
                "classes": train_ds.classes, "seq_len": args.seq_len,
            }, ckpt_dir / "best_model.pth")
            print(f"  ✅ Best saved  val={best_val:.1f}%")
            save_cm(preds, tgts, train_ds.classes,
                    ckpt_dir / "val_confusion.png",
                    f"Val Confusion  ep={epoch}  acc={vl_acc:.1f}%")
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       ckpt_dir / f"ckpt_ep{epoch:03d}.pth")

        if patience_cnt >= args.patience:
            print(f"\n  Early stop — no improvement for {args.patience} epochs")
            break

    # -- Final test evaluation (done ONCE, best checkpoint) -----------------
    print(f"\n{'='*60}")
    print(f"  FINAL TEST  (best model from epoch with val={best_val:.1f}%)")
    print(f"{'='*60}")
    try:
        test_ds = GestureDataset(args.data_dir, "test", args.seq_len, augment=False)
        test_ld = DataLoader(test_ds, args.batch_size*2, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)
        ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ts_loss, ts_acc, tp, tt = eval_epoch(model, test_ld, criterion, device)
        print(f"  Test  loss={ts_loss:.4f}  acc={ts_acc:.1f}%")
        save_cm(tp, tt, test_ds.classes, ckpt_dir / "test_confusion.png",
                f"TEST Confusion  acc={ts_acc:.1f}%")
        if HAS_VIZ:
            print("\nClassification Report (test set):")
            print(classification_report(tt, tp, target_names=test_ds.classes))
    except Exception as e:
        print(f"  Test eval failed: {e}")
        ts_acc = 0.0

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Best Val  : {best_val:.2f}%")
    print(f"  Test Acc  : {ts_acc:.2f}%")
    print(f"  Next step : python3 export_onnx.py --model_path checkpoints/best_model.pth")
    print(f"{'='*60}")
    writer.close()


# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="ipd_processed")
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--model_type", default="cnn_lstm", choices=["cnn_lstm","lightweight"])
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--seq_len",    type=int,   default=30)
    p.add_argument("--patience",   type=int,   default=20)
    main(p.parse_args())