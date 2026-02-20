#!/usr/bin/env python3
"""
Pre-flight Check — Run this BEFORE processing the dataset.
Checks: GPU, trt_pose, MediaPipe, dataset files, annotation files, output scripts.
"""

import os, sys, json
from pathlib import Path

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

results = []

def check(label, ok, detail="", fatal=False):
    sym = PASS if ok else (FAIL if fatal else WARN)
    line = f"{sym}  {label}"
    if detail:
        line += f"\n        {detail}"
    print(line)
    results.append((ok, fatal, label))
    return ok

print("\n" + "="*60)
print("  PRE-FLIGHT DIAGNOSTIC")
print("="*60)

# ── 1. Python & PyTorch ─────────────────────────────────────
print("\n[1] Python / PyTorch / CUDA")
check("Python version", sys.version_info >= (3,7),
      f"Python {sys.version.split()[0]}", fatal=True)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    check("PyTorch installed", True, f"v{torch.__version__}")
    check("CUDA available", cuda_ok,
          f"GPU: {gpu_name}" if cuda_ok else "No CUDA GPU found!", fatal=True)
    if cuda_ok:
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU memory", mem >= 3.0, f"{mem:.1f} GB VRAM")
except ImportError:
    check("PyTorch installed", False, "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", fatal=True)

# ── 2. trt_pose ─────────────────────────────────────────────
print("\n[2] trt_pose_hand")
try:
    import trt_pose
    check("trt_pose installed", True, "GPU keypoint extraction available")
    TRT_POSE_OK = True
except ImportError:
    check("trt_pose installed", False,
          "Will use MediaPipe fallback (OK for now)\n"
          "        To install: bash setup_trtpose.sh\n"
          "        Or: git clone https://github.com/NVIDIA-AI-IOT/trt_pose && cd trt_pose && python setup.py install")
    TRT_POSE_OK = False

try:
    import torch2trt
    check("torch2trt installed", True, "TensorRT conversion available")
except ImportError:
    check("torch2trt installed", False,
          "git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install")

# ── 3. trt_pose model weights ────────────────────────────────
print("\n[3] trt_pose Model Weights")
weight_files = [
    "hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth",
    "hand_pose_densenet121_baseline_att_224x224_B_epoch_149.pth",
]
found_weights = []
for wf in weight_files:
    exists = Path(wf).exists()
    if exists:
        sz = Path(wf).stat().st_size / 1e6
        check(wf[:30]+"...", True, f"{sz:.0f} MB")
        found_weights.append(wf)
    else:
        check(wf[:30]+"...", False,
              "Download from https://github.com/NVIDIA-AI-IOT/trt_pose_hand\n"
              "        or: pip install gdown && gdown 'https://drive.google.com/uc?id=1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo'")

topology_paths = [
    "preprocess/hand_pose.json",
    "hand_pose.json",
]
found_topo = any(Path(p).exists() for p in topology_paths)
if found_topo:
    found_p = next(p for p in topology_paths if Path(p).exists())
    check("hand_pose.json (topology)", True, found_p)
else:
    check("hand_pose.json (topology)", False,
          "curl -L https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose_hand/master/preprocess/hand_pose.json -o preprocess/hand_pose.json\n"
          "        mkdir -p preprocess && curl -L <url> -o preprocess/hand_pose.json")

# ── 4. MediaPipe fallback ────────────────────────────────────
print("\n[4] MediaPipe (fallback extractor)")
try:
    import mediapipe as mp
    check("MediaPipe installed", True, "Fallback available if trt_pose fails")
except ImportError:
    check("MediaPipe installed", False,
          "pip install mediapipe  ← needed if trt_pose not available")

# ── 5. IPD Dataset ──────────────────────────────────────────
print("\n[5] IPD Dataset")

# Common paths to check (WSL2, native Linux, etc.)
DATASET_CANDIDATES = [
    "/mnt/c/Users/Soham/Downloads/IPD dataset",
    "/mnt/c/Users/babatunde/Downloads/IPD dataset",
    os.path.expanduser("~/IPD dataset"),
    os.path.expanduser("~/Downloads/IPD dataset"),
    "./IPD dataset",
    "./ipd_dataset",
]
# Also check env var
if "IPD_DIR" in os.environ:
    DATASET_CANDIDATES.insert(0, os.environ["IPD_DIR"])

ipd_root = None
for c in DATASET_CANDIDATES:
    if Path(c).exists():
        ipd_root = Path(c)
        break

if ipd_root:
    check("IPD dataset root found", True, str(ipd_root))

    # Check videos
    video_dirs = [ipd_root/"videos"/"videos", ipd_root/"videos", ipd_root]
    video_dir = None
    for vd in video_dirs:
        avi_files = list(vd.glob("*.avi")) if vd.exists() else []
        if avi_files:
            video_dir = vd
            check(f"Videos (.avi) found", True, f"{len(avi_files)} files in {vd}")
            break
    if not video_dir:
        check("Videos (.avi) found", False,
              f"Looked in: {[str(d) for d in video_dirs]}", fatal=True)

    # Check annotations
    annot_candidates = [
        ipd_root/"annotations"/"Annot_TrainList.txt",
        ipd_root/"annotations"/"annotations"/"Annot_TrainList.txt",
        ipd_root/"Annot_TrainList.txt",
    ]
    annot_file = None
    for ac in annot_candidates:
        if ac.exists():
            annot_file = ac
            break

    if annot_file:
        # Count gesture lines
        lines = [l for l in annot_file.read_text().splitlines()
                 if l.strip() and not l.startswith('#')]
        target = {'G01','G02','G05','G06','G10','G11'}
        target_lines = [l for l in lines if len(l.split(',')) > 1 and l.split(',')[1].strip() in target]
        check("Annot_TrainList.txt found", True,
              f"{annot_file}\n        {len(lines)} total annotations, "
              f"{len(target_lines)} for target gestures (G01/G02/G05/G06/G10/G11)")
    else:
        check("Annot_TrainList.txt found", False,
              f"Looked in: {[str(a) for a in annot_candidates]}", fatal=True)

    # Check class_details
    class_details_candidates = [
        ipd_root/"annotations"/"class_details.txt",
        ipd_root/"class_details.txt",
    ]
    found_cd = any(Path(p).exists() for p in class_details_candidates)
    check("class_details.txt found", found_cd,
          "Optional but useful for verifying gesture codes")

else:
    check("IPD dataset root found", False,
          f"Tried: {DATASET_CANDIDATES[:4]}\n"
          "        Set path with: export IPD_DIR='/path/to/IPD dataset'\n"
          "        OR use: python3 process_ipd_trtpose.py --ipd_dir '/your/path'",
          fatal=True)

# ── 6. Other Python packages ─────────────────────────────────
print("\n[6] Python packages")
packages = {
    'cv2':           ('opencv-python', 'pip install opencv-python'),
    'numpy':         ('numpy',         'pip install numpy'),
    'tqdm':          ('tqdm',          'pip install tqdm'),
    'sklearn':       ('scikit-learn',  'pip install scikit-learn'),
    'tensorboard':   ('tensorboard',   'pip install tensorboard'),
    'seaborn':       ('seaborn',       'pip install seaborn'),
    'onnx':          ('onnx',          'pip install onnx'),
    'onnxruntime':   ('onnxruntime',   'pip install onnxruntime'),
    'matplotlib':    ('matplotlib',    'pip install matplotlib'),
}
missing = []
for mod, (pkg, install) in packages.items():
    try:
        __import__(mod)
        check(pkg, True)
    except ImportError:
        check(pkg, False, install)
        missing.append(install)

# ── 7. Project scripts ──────────────────────────────────────
print("\n[7] Project Scripts")
scripts = [
    "models.py",
    "train.py",
    "process_ipd_trtpose.py",
    "export_onnx.py",
    "jetson_inference_trtpose.py",
]
for s in scripts:
    check(s, Path(s).exists(),
          f"Download from your chat outputs" if not Path(s).exists() else "")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "="*60)
fatals  = [l for ok, fatal, l in results if not ok and fatal]
warns   = [l for ok, fatal, l in results if not ok and not fatal]
passing = [l for ok, _, l in results if ok]

print(f"  Passed : {len(passing)}")
print(f"  Warnings: {len(warns)}")
print(f"  FATAL  : {len(fatals)}")

if fatals:
    print(f"\n  🔴 Fix these FIRST:")
    for f in fatals: print(f"     • {f}")
elif warns:
    print(f"\n  🟡 Optional fixes:")
    for w in warns: print(f"     • {w}")
    print(f"\n  ✅ You can proceed with warnings — use fallback mode.")
else:
    print(f"\n  🟢 Everything ready! Run:")
    if TRT_POSE_OK:
        print(f"     python3 process_ipd_trtpose.py \\")
        print(f"         --ipd_dir '{ipd_root}' \\")
        print(f"         --output_dir ipd_processed \\")
        print(f"         --max_videos 5")
    else:
        print(f"     python3 process_ipd_trtpose.py \\")
        print(f"         --ipd_dir '{ipd_root}' \\")
        print(f"         --output_dir ipd_processed \\")
        print(f"         --max_videos 5")
        print(f"     (Will use MediaPipe fallback)")

# ── Exact command for their setup ───────────────────────────
print("\n" + "="*60)
print(" YOUR NEXT COMMAND (copy-paste):")
print("="*60)
if ipd_root:
    print(f"""
  # Quick test (5 videos, ~2 min):
  python3 process_ipd_trtpose.py \\
      --ipd_dir "{ipd_root}" \\
      --output_dir ipd_processed \\
      --seq_len 30 \\
      --max_videos 5

  # If test passes, full dataset:
  python3 process_ipd_trtpose.py \\
      --ipd_dir "{ipd_root}" \\
      --output_dir ipd_processed \\
      --seq_len 30
""")
else:
    print("""
  # Set your IPD dataset path first:
  export IPD_DIR="/mnt/c/Users/Soham/Downloads/IPD dataset"

  # Then test with 5 videos:
  python3 process_ipd_trtpose.py \\
      --ipd_dir "$IPD_DIR" \\
      --output_dir ipd_processed \\
      --seq_len 30 \\
      --max_videos 5
""")
