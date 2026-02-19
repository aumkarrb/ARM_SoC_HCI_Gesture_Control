"""
process_frames.py — Raw frame extractor for IPN Hand dataset
============================================================
Replaces process_ipd_trtpose.py (keypoints) with raw RGB frames.
Saves (seq_len, 112, 112, 3) uint8 arrays → ~1.1MB per sequence.
No MediaPipe. No keypoints. Just frames.

Usage:
  python3 process_frames.py \
      --ipd_dir  "/mnt/c/Users/Soham/Desktop/Final/merged_dataset" \
      --output_dir frame_processed \
      --seq_len 16 \
      --stride 8 \
      --img_size 112
"""

import os, json, random, argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

TARGET_GESTURES = {
    'G01': 'click_one',
    'G02': 'click_two',
    'G05': 'throw_left',
    'G06': 'throw_right',
    'G10': 'zoom_in',
    'G11': 'zoom_out',
}


def parse_annotations(annot_path, target_gestures):
    annotations = defaultdict(list)
    target_set  = set(target_gestures)
    with open(annot_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 5: continue
            video = parts[0].strip()
            code  = parts[1].strip()
            try:
                sf, ef = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            if code in target_set:
                annotations[video].append((code, sf, ef))
    return dict(annotations)


def extract_frames_sliding_window(cap, start_frame, end_frame,
                                   seq_len, stride, img_size):
    """
    Extract raw frames as sliding windows from one annotation segment.
    Returns list of (seq_len, img_size, img_size, 3) uint8 arrays.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    all_frames = []

    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        # Square crop from center (eliminates NORM_RECT distortion at source)
        h, w = frame.shape[:2]
        size = min(h, w)
        x0   = (w - size) // 2
        y0   = (h - size) // 2
        frame = frame[y0:y0+size, x0:x0+size]
        frame = cv2.resize(frame, (img_size, img_size),
                           interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

    total = len(all_frames)
    if total < seq_len:
        return []

    sequences = []
    for ws in range(0, total - seq_len + 1, stride):
        window = all_frames[ws : ws + seq_len]
        seq    = np.array(window, dtype=np.uint8)  # (seq_len, H, W, 3)
        sequences.append(seq)

    return sequences


def process_dataset(ipd_dir, output_dir, seq_len=16, stride=8,
                    img_size=112, train_ratio=0.70, val_ratio=0.10):

    ipd_path    = Path(ipd_dir)
    output_path = Path(output_dir)
    test_ratio  = round(1.0 - train_ratio - val_ratio, 2)

    print("\n" + "="*65)
    print("  Frame Dataset Extractor")
    print("="*65)
    print(f"  Source  : {ipd_path}")
    print(f"  Output  : {output_path}")
    print(f"  Seq len : {seq_len} frames")
    print(f"  Stride  : {stride} frames")
    print(f"  Size    : {img_size}x{img_size} px")
    print(f"  Split   : {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")
    print("="*65 + "\n")

    # Find annotation file
    annot_file = None
    for c in [ipd_path/'annotations'/'annotations'/'Annot_TrainList.txt',
              ipd_path/'annotations'/'Annot_TrainList.txt',
              ipd_path/'Annot_TrainList.txt']:
        if c.exists(): annot_file = c; break
    if not annot_file:
        raise FileNotFoundError(f"Annot_TrainList.txt not found under {ipd_path}")

    annotations = parse_annotations(str(annot_file), list(TARGET_GESTURES.keys()))
    print(f"  Annotated videos: {len(annotations)}")

    # Find video dir
    video_dir = None
    for c in [ipd_path/'videos'/'videos', ipd_path/'videos', ipd_path]:
        if c.exists() and any(c.glob('*.avi')):
            video_dir = c; break
    if not video_dir:
        raise FileNotFoundError(f"No .avi files found under {ipd_path}")
    print(f"  Video dir : {video_dir}")

    # Split videos by name (not by sequence)
    all_videos = sorted(annotations.keys())
    random.seed(42)
    random.shuffle(all_videos)
    n = len(all_videos)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    split_map = (
        {v: 'train' for v in all_videos[:n_train]}    |
        {v: 'val'   for v in all_videos[n_train:n_train+n_val]} |
        {v: 'test'  for v in all_videos[n_train+n_val:]}
    )

    # Create output dirs
    for split in ('train', 'val', 'test'):
        for cls in TARGET_GESTURES.values():
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    counters = {sp: defaultdict(int) for sp in ('train','val','test')}

    # Process each video
    for vi, vname in enumerate(sorted(annotations.keys())):
        split = split_map[vname]
        vpath = video_dir / f"{vname}.avi"
        if not vpath.exists():
            print(f"  ⚠  Missing: {vname}.avi")
            continue

        cap    = cv2.VideoCapture(str(vpath))
        n_good = 0
        for code, sf, ef in annotations[vname]:
            cls  = TARGET_GESTURES[code]
            seqs = extract_frames_sliding_window(
                cap, sf, ef, seq_len, stride, img_size)
            for seq in seqs:
                i = counters[split][cls]
                np.save(output_path / split / cls / f"seq_{i:05d}.npy", seq)
                counters[split][cls] += 1
                n_good += 1
        cap.release()

        total = sum(sum(c.values()) for c in counters.values())
        print(f"  [{vi+1:3d}/{n}] {vname:30s} [{split:5s}] +{n_good:<3d} total={total}")

    # Summary
    print("\n")
    print(f"  {'Class':15s}  {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print("  " + "-"*38)
    for cls in sorted(TARGET_GESTURES.values()):
        tr = counters['train'][cls]
        va = counters['val'][cls]
        te = counters['test'][cls]
        print(f"  {cls:15s}  {tr:6d}  {va:6d}  {te:6d}")
    print("  " + "-"*38)
    tr_t = sum(counters['train'].values())
    va_t = sum(counters['val'].values())
    te_t = sum(counters['test'].values())
    print(f"  {'TOTAL':15s}  {tr_t:6d}  {va_t:6d}  {te_t:6d}")

    per = tr_t // max(len(TARGET_GESTURES), 1)
    if per >= 100:
        print(f"\n  ✅ {tr_t} train seqs (~{per}/class) → CNN+LSTM ready")
    else:
        print(f"\n  ⚠️  Only {per}/class — consider smaller stride")

    # Save metadata
    meta = {
        'type': 'frames',
        'seq_len': seq_len, 'stride': stride, 'img_size': img_size,
        'classes': sorted(TARGET_GESTURES.values()),
        'counts': {sp: dict(counters[sp]) for sp in ('train','val','test')},
    }
    json.dump(meta, open(output_path/'metadata.json','w'), indent=2)

    print(f"\n{'='*65}")
    print(f"  ✅ Done!  Next step:")
    print(f"  python3 train_frames.py --data_dir {output_dir}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ipd_dir',    required=True)
    p.add_argument('--output_dir', default='frame_processed')
    p.add_argument('--seq_len',    type=int, default=16,
                   help='Frames per sequence (16 = ~0.5s at 30fps)')
    p.add_argument('--stride',     type=int, default=8,
                   help='Sliding window stride (8 = 50% overlap)')
    p.add_argument('--img_size',   type=int, default=112,
                   help='Square frame size (112 is standard, 224 is slower)')
    p.add_argument('--train_ratio',type=float, default=0.70)
    p.add_argument('--val_ratio',  type=float, default=0.10)
    args = p.parse_args()
    process_dataset(
        ipd_dir=args.ipd_dir, output_dir=args.output_dir,
        seq_len=args.seq_len, stride=args.stride,
        img_size=args.img_size,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )