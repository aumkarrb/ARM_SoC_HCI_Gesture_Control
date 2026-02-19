#!/usr/bin/env python3
"""
IPD Dataset Processor — Final Version
======================================
Extractor priority (auto-detected, no manual config):
  1. trt_pose_hand (GPU)  — fastest, needs weights + hand_pose.json
  2. MediaPipe ≤0.9       — reliable legacy API (pip install mediapipe==0.9.3)
  3. MediaPipe ≥0.10      — new Tasks API (needs hand_landmarker.task file)

KEY FIX vs all previous versions:
  mediapipe is imported at module level (top of file, unconditionally).
  Previous versions only imported it inside the trt_pose except-block,
  so when trt_pose loaded but weights were missing, mp was never defined.

Split: 70% train / 10% val / 20% test — split BY VIDEO (no data leakage)
Output: ipd_processed/train/ val/ test/  each with one folder per gesture class
"""

import os, sys, cv2, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("ERROR: pip install torch torchvision"); sys.exit(1)

# ── MediaPipe — ALWAYS imported at module level ───────────────────────────────
# This is the root cause of the NameError in all previous versions.
# trt_pose may succeed but then fail to find weights and fall back to MediaPipe.
# If mp was only imported inside `except ImportError`, it's not in scope here.
# Solution: import it unconditionally at the top, set flags for which API is live.

mp        = None   # mediapipe module (or None if not installed)
mp_vision = None   # mediapipe.tasks.python.vision  (new API only)
mp_base   = None   # mediapipe.tasks.python.core.base_options (new API only)
MP_OK      = False  # True = some usable MediaPipe API found
MP_LEGACY  = False  # True = old solutions.hands API (mediapipe ≤ 0.9)
MP_NEW     = False  # True = new Tasks API (mediapipe ≥ 0.10)

try:
    import mediapipe as mp                    # ← at module level, always
    try:
        _test = mp.solutions.hands            # exists in ≤ 0.9
        MP_OK     = True
        MP_LEGACY = True
    except AttributeError:
        try:                                  # ≥ 0.10 removed solutions
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core import base_options as mp_base
            MP_OK  = True
            MP_NEW = True
        except ImportError:
            pass
except ImportError:
    pass   # no mediapipe — fine if trt_pose works with its weights

# ── trt_pose — imported after mediapipe so fallback is always ready ───────────
TRT_POSE_OK = False
try:
    import trt_pose.coco, trt_pose.models
    from trt_pose.parse_objects import ParseObjects
    TRT_POSE_OK = True
    print("✅ trt_pose_hand loaded (GPU)")
except ImportError:
    if MP_OK:
        api = "legacy ≤0.9" if MP_LEGACY else "new API ≥0.10"
        print(f"⚠️  trt_pose not installed → MediaPipe ({api}) will be used")
    else:
        print("ERROR: Neither trt_pose nor MediaPipe available.")
        print("  Fix A (GPU) : bash install_trtpose.sh")
        print("  Fix B (CPU) : pip install 'mediapipe==0.9.3'")
        sys.exit(1)

# ── Target gestures ───────────────────────────────────────────────────────────
TARGET_GESTURES = {
    'G01': 'click_one',    # Click with one finger   → Play/Pause
    'G02': 'click_two',    # Click with two fingers  → Stop
    'G05': 'throw_left',   # Throw left              → Previous Track
    'G06': 'throw_right',  # Throw right             → Next Track
    'G10': 'zoom_in',      # Zoom in                 → Volume Up
    'G11': 'zoom_out',     # Zoom out                → Volume Down
}


# ─────────────────────────────────────────────────────────────────────────────
#  trt_pose_hand extractor  (GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TrtPoseHandExtractor:
    """
    GPU hand-keypoint extraction via NVIDIA trt_pose_hand.

    Needs two files in ~/gesture_final/:
      preprocess/hand_pose.json                             (topology)
      hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth  (weights)

    Get them:
      curl -L https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose_hand/
           master/preprocess/hand_pose.json -o preprocess/hand_pose.json
      # weights: download manually from browser (see instructions below)
    """
    WEIGHTS = {
        'resnet18':    'hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth',
        'densenet121': 'hand_pose_densenet121_baseline_att_224x224_B_epoch_149.pth',
    }
    MEAN = torch.Tensor([0.485, 0.456, 0.406])
    STD  = torch.Tensor([0.229, 0.224, 0.225])

    def __init__(self, model_type='resnet18', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mean   = self.MEAN.to(self.device)[:, None, None]
        self.std    = self.STD.to(self.device)[:, None, None]

        topo = self._find('hand_pose.json', [
            'preprocess/hand_pose.json',
            'hand_pose.json',
            os.path.expanduser('~/trt_pose/tasks/hand_pose/preprocess/hand_pose.json'),
            os.path.expanduser('~/gesture_final/preprocess/hand_pose.json'),
        ])
        with open(topo) as f:
            hp = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(hp)
        n_parts = len(hp['keypoints'])   # 21
        n_links = len(hp['skeleton'])

        wf = self._find(self.WEIGHTS[model_type], [
            self.WEIGHTS[model_type],
            os.path.expanduser(f'~/{self.WEIGHTS[model_type]}'),
            os.path.expanduser(f'~/gesture_final/{self.WEIGHTS[model_type]}'),
        ])

        if model_type == 'resnet18':
            self.model = trt_pose.models.resnet18_baseline_att(
                n_parts, 2*n_links).to(self.device).eval()
        else:
            self.model = trt_pose.models.densenet121_baseline_att(
                n_parts, 2*n_links).to(self.device).eval()
        self.model.load_state_dict(torch.load(wf, map_location=self.device))
        self.parse = ParseObjects(self.topology,
                                  cmap_threshold=0.15, link_threshold=0.15)
        print(f"  ✅ trt_pose ({model_type}, {self.device})")

    def _find(self, name, candidates):
        for c in candidates:
            if c and os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"{name} not found.\n"
            f"  Tried: {[c for c in candidates if c]}"
        )

    @torch.no_grad()
    def extract(self, bgr):
        img = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t   = ((torch.from_numpy(img).float().to(self.device) / 255.0)
               .permute(2, 0, 1) - self.mean) / self.std
        cmap, paf = self.model(t.unsqueeze(0))
        counts, objects, peaks = self.parse(cmap.cpu(), paf.cpu())
        if counts[0] == 0:
            return None
        obj = objects[0][0]
        kp  = np.zeros((21, 2), dtype=np.float32)
        for i in range(21):
            k = int(obj[i])
            if k >= 0:
                kp[i] = [peaks[0][i][k][1].item(), peaks[0][i][k][0].item()]
        return kp.flatten()


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe extractor  (CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MediaPipeHandExtractor:
    """
    CPU fallback extractor.
    mp is imported at module level so this always works regardless of
    whether trt_pose loaded or not. Supports both API versions.
    """
    def __init__(self):
        if not MP_OK:
            raise RuntimeError(
                "MediaPipe unavailable. Fix: pip install 'mediapipe==0.9.3'"
            )
        if MP_LEGACY:
            # ── mediapipe ≤ 0.9  (mp.solutions.hands) ──────────────
            self.hands = mp.solutions.hands.Hands(   # mp is the module-level import
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mode = 'legacy'
            print("  ✅ MediaPipe extractor (legacy API ≤0.9)")
        else:
            # ── mediapipe ≥ 0.10  (HandLandmarker Tasks API) ────────
            task = self._find_task()
            self.landmarker = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_base.BaseOptions(model_asset_path=task),
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    running_mode=mp_vision.RunningMode.VIDEO,
                )
            )
            self._mode     = 'new'
            self._frame_ts = 0
            print(f"  ✅ MediaPipe extractor (new Tasks API ≥0.10)")

    def _find_task(self):
        # Search common locations
        candidates = [
            'hand_landmarker.task',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task'),
            os.path.expanduser('~/gesture_final/hand_landmarker.task'),
            os.path.expanduser('~/hand_landmarker.task'),
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"  Found hand_landmarker.task: {c}")
                return c

        # Auto-download if not found
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task'
        )
        print("  hand_landmarker.task not found — downloading automatically (~30MB)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, save_path,
                reporthook=lambda b, bs, t: print(
                    f"  Downloading... {min(100, int(b*bs/t*100))}%",
                    end='\r') if t > 0 else None)
            print(f"\n  ✅ Downloaded → {save_path}")
            return save_path
        except Exception as e:
            raise FileNotFoundError(
                f"Auto-download failed: {e}\n"
                "Manual download:\n"
                f"  curl -L '{url}' -o hand_landmarker.task"
            )

    def extract(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._mode == 'legacy':
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            kp = np.array([[p.x, p.y]
                            for p in res.multi_hand_landmarks[0].landmark],
                           dtype=np.float32)
        else:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_ts += 33
            res = self.landmarker.detect_for_video(img, self._frame_ts)
            if not res.hand_landmarks:
                return None
            kp = np.array([[lm.x, lm.y] for lm in res.hand_landmarks[0]],
                          dtype=np.float32)
        return kp.flatten()

    def __del__(self):
        if hasattr(self, 'hands'):      self.hands.close()
        if hasattr(self, 'landmarker'): self.landmarker.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Normalization — identical in processor, train.py, and inference
# ─────────────────────────────────────────────────────────────────────────────

def normalize_keypoints(kp_flat):
    """
    1. Center on wrist (landmark 0) — removes hand position in frame
    2. Scale by wrist→middle-tip distance — removes hand size variation
    Result: only finger shape and motion survive → gesture-relevant features only
    """
    kp = kp_flat.reshape(21, 2).copy()
    kp -= kp[0]
    scale = np.linalg.norm(kp[12]) + 1e-6
    kp   /= scale
    return kp.flatten()


# ─────────────────────────────────────────────────────────────────────────────
#  Annotation parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_annot_trainlist(annot_path, target_gestures):
    annotations = defaultdict(list)
    target_set  = set(target_gestures)
    with open(annot_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split(',')
            if len(parts) < 5: continue
            video, code = parts[0].strip(), parts[1].strip()
            try:
                sf, ef = int(parts[3]), int(parts[4])
            except ValueError:
                continue
            if code in target_set:
                annotations[video].append((code, sf, ef))
    print(f"  Parsed {len(annotations)} relevant videos")
    return dict(annotations)


# ─────────────────────────────────────────────────────────────────────────────
#  Sliding window extractor  (replaces single-sequence extractor)
# ─────────────────────────────────────────────────────────────────────────────

def extract_sliding_window(cap, start_frame, end_frame, extractor, seq_len, stride):
    """
    Extract ALL sequences from one annotation segment using a sliding window.

    Old: 1 segment → 1 sequence (30 frames downsampled from 137)
    New: 1 segment → N sequences (each 30 real consecutive frames, stride apart)

    Example — 137-frame segment, seq_len=30, stride=10:
      Window 0: frames 0-29     → sequence 1
      Window 1: frames 10-39    → sequence 2
      ...
      Window 10: frames 100-129 → sequence 11   (11x more real data!)

    Windows with <60% hand detection are discarded (quality filter).
    Missing frames are forward-filled from the last detected keypoint.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    all_kp = []

    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        kp = extractor.extract(frame)
        all_kp.append(normalize_keypoints(kp) if kp is not None else None)

    total = len(all_kp)
    if total < seq_len:
        return []

    sequences = []
    for ws in range(0, total - seq_len + 1, stride):
        window   = all_kp[ws : ws + seq_len]
        detected = [f for f in window if f is not None]
        if len(detected) / seq_len < 0.60:
            continue                          # too many missed frames
        last = detected[0]
        filled = []
        for f in window:
            if f is not None: last = f
            filled.append(last)              # forward-fill gaps
        sequences.append(np.array(filled, dtype=np.float32))

    return sequences


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_ipd_dataset(ipd_dir, output_dir, seq_len=30, stride=10,
                        target_gestures=None, max_videos=None,
                        model_type='resnet18',
                        train_ratio=0.70, val_ratio=0.10):

    assert abs(train_ratio + val_ratio - 0.80) < 0.001, \
        "train_ratio + val_ratio must equal 0.80 (test=0.20)"

    ipd_path    = Path(ipd_dir)
    output_path = Path(output_dir)
    target_gest = target_gestures or list(TARGET_GESTURES.keys())
    test_ratio  = round(1.0 - train_ratio - val_ratio, 2)

    # ── Choose extractor ─────────────────────────────────────────────────
    extractor      = None
    extractor_name = None

    if TRT_POSE_OK:
        try:
            extractor      = TrtPoseHandExtractor(model_type=model_type)
            extractor_name = 'trt_pose_hand'
        except FileNotFoundError as e:
            print(f"\n  ⚠️  trt_pose file missing: {e}")
            print("  → Falling back to MediaPipe\n")

    if extractor is None:
        extractor      = MediaPipeHandExtractor()   # mp is module-level → always works
        extractor_name = 'mediapipe'

    print("\n" + "="*65)
    print("  IPD Dataset Processor — Final Version")
    print("="*65)
    print(f"  IPD dir   : {ipd_path}")
    print(f"  Output    : {output_path}")
    print(f"  Extractor : {extractor_name}")
    print(f"  Seq len   : {seq_len} frames")
    print(f"  Stride    : {stride} frames  (sliding window → more sequences per segment)")
    print(f"  Split     : train={train_ratio:.0%}  val={val_ratio:.0%}  test={test_ratio:.0%}  (by video)")
    print(f"  Gestures  : {target_gest}")
    print("="*65 + "\n")

    # ── Find annotation file ─────────────────────────────────────────────
    annot_file = None
    for c in [ipd_path/'annotations'/'annotations'/'Annot_TrainList.txt',
              ipd_path/'annotations'/'Annot_TrainList.txt',
              ipd_path/'Annot_TrainList.txt']:
        if c.exists(): annot_file = c; break
    if not annot_file:
        raise FileNotFoundError(f"Annot_TrainList.txt not found under {ipd_path}")
    annotations = parse_annot_trainlist(str(annot_file), target_gest)

    # ── Find video dir ────────────────────────────────────────────────────
    video_dir = None
    for c in [ipd_path/'videos'/'videos', ipd_path/'videos', ipd_path]:
        if c.exists() and list(c.glob('*.avi')):
            video_dir = c; break
    if not video_dir:
        raise FileNotFoundError(f"No .avi files under {ipd_path}")
    print(f"  Video dir : {video_dir}\n")

    # ── Video-level split ─────────────────────────────────────────────────
    all_videos = sorted(annotations.keys())
    if max_videos:
        all_videos = all_videos[:max_videos]

    rng = np.random.default_rng(42)
    arr = np.array(all_videos)
    rng.shuffle(arr)

    n    = len(arr)
    n_tr = max(1, int(n * train_ratio))
    n_vl = max(1, int(n * val_ratio))

    video_splits = {
        'train': set(arr[:n_tr]),
        'val':   set(arr[n_tr:n_tr+n_vl]),
        'test':  set(arr[n_tr+n_vl:]),
    }
    print("  Videos per split:")
    for sp, vs in video_splits.items():
        print(f"    {sp:5s}: {len(vs)}")
    print()

    # ── Create output dirs ────────────────────────────────────────────────
    classes = sorted({TARGET_GESTURES[g] for g in target_gest})
    for split in ('train', 'val', 'test'):
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    # ── Extract ───────────────────────────────────────────────────────────
    counters = defaultdict(lambda: defaultdict(int))
    total    = len(arr)

    for vi, vname in enumerate(arr):
        split = next(sp for sp, vs in video_splits.items() if vname in vs)
        vfile = video_dir / f"{vname}.avi"
        if not vfile.exists():
            m = list(video_dir.glob(f"{vname}*"))
            if not m: print(f"\n  ⚠  Missing: {vname}"); continue
            vfile = m[0]

        cap = cv2.VideoCapture(str(vfile))
        if not cap.isOpened(): continue

        n_good = 0
        for code, sf, ef in annotations[vname]:
            cls  = TARGET_GESTURES[code]
            seqs = extract_sliding_window(cap, sf, ef, extractor, seq_len, stride)
            for seq in seqs:
                i = counters[split][cls]
                np.save(output_path / split / cls / f"seq_{i:05d}.npy", seq)
                counters[split][cls] += 1
                n_good += 1
        cap.release()

        done = sum(sum(d.values()) for d in counters.values())
        print(f"  [{vi+1:3d}/{total}] {vname[:38]:38s} [{split:5s}] +{n_good}  total={done}",
              end='\r')

    print("\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"  {'Class':15s}  {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print("  " + "-"*38)
    for cls in classes:
        tr = counters['train'].get(cls, 0)
        vl = counters['val'].get(cls, 0)
        ts = counters['test'].get(cls, 0)
        print(f"  {cls:15s}  {tr:6d}  {vl:6d}  {ts:6d}")
    totals = [sum(counters[sp].values()) for sp in ('train','val','test')]
    print("  " + "-"*38)
    print(f"  {'TOTAL':15s}  {totals[0]:6d}  {totals[1]:6d}  {totals[2]:6d}")

    per_class = totals[0] // max(len(classes), 1)
    print()
    if totals[0] < 400:
        print(f"  ⚠️  Small dataset (~{per_class}/class) → train.py auto-uses LightweightCNNLSTM")
    else:
        print(f"  ✅ {totals[0]} train seqs (~{per_class}/class) → full GestureCNNLSTM")

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'extractor':    extractor_name,
            'features':     42, 'seq_len': seq_len,
            'norm':         'center_wrist_scale_middle_tip',
            'classes':      classes,
            'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
            'split_by':     'video',
            'counts':       {sp: dict(counters[sp]) for sp in ('train','val','test')},
        }, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✅  Done!  Extractor: {extractor_name}")
    print(f"{'='*65}")
    print(f"\n  Next step:")
    print(f"  python3 train.py --data_dir {output_dir} --epochs 100 --batch_size 32\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ipd_dir',     required=True)
    p.add_argument('--output_dir',  default='ipd_processed')
    p.add_argument('--seq_len',     type=int,   default=30)
    p.add_argument('--gestures',    nargs='+',  default=None)
    p.add_argument('--max_videos',  type=int,   default=None)
    p.add_argument('--model',       default='resnet18',
                   choices=['resnet18','densenet121'])
    p.add_argument('--stride',      type=int,   default=10,
                   help='Sliding window stride in frames (smaller=more sequences, default=10)')
    p.add_argument('--train_ratio', type=float, default=0.70)
    p.add_argument('--val_ratio',   type=float, default=0.10)
    args = p.parse_args()
    process_ipd_dataset(
        ipd_dir         = args.ipd_dir,
        output_dir      = args.output_dir,
        seq_len         = args.seq_len,
        stride          = args.stride,
        target_gestures = args.gestures,
        max_videos      = args.max_videos,
        model_type      = args.model,
        train_ratio     = args.train_ratio,
        val_ratio       = args.val_ratio,
    )