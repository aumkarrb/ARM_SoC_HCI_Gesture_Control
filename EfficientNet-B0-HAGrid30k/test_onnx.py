"""
test_onnx.py — Verify gesture_model.onnx works correctly
=========================================================
Tests the ONNX model on 5 images per class and compares
output against the PyTorch model. If they match, ONNX is correct.

Usage:
  python3 test_onnx.py
"""

import numpy as np
import torch
import torchvision.models as tvm
import torchvision.transforms as T
import torch.nn as nn
from pathlib import Path
from PIL import Image
import json

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR  = Path('checkpoints_hagrid')
DATA_DIR  = Path("/mnt/c/Users/Soham/Downloads/Datasets/hagrid-sample-30k-384p/hagrid_30k")
ONNX_PATH = CKPT_DIR / 'gesture_model.onnx'

# ── Load classes ──────────────────────────────────────────────────────────────
ckpt    = torch.load(CKPT_DIR / 'best_model.pth', map_location='cpu', weights_only=False)
classes = ckpt['classes']
print(f"Classes: {classes}")

# ── Load PyTorch model ────────────────────────────────────────────────────────
pt_model = tvm.efficientnet_b0(weights=None)
pt_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(pt_model.classifier[1].in_features, len(classes))
)
pt_model.load_state_dict(ckpt['model_state_dict'])
pt_model.eval()
print("✅ PyTorch model loaded")

# ── Load ONNX model ───────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=['CUDAExecutionProvider',
                                           'CPUExecutionProvider'])
    print(f"✅ ONNX model loaded  ({ONNX_PATH})")
    print(f"   Provider: {sess.get_providers()[0]}")
except ImportError:
    print("❌ onnxruntime not installed")
    print("   Run: pip install onnxruntime --break-system-packages")
    exit(1)

# ── Transform ─────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Test ──────────────────────────────────────────────────────────────────────
print(f"\n{'Class':20s} {'PT pred':20s} {'ONNX pred':20s} {'Match':6s} {'Conf':8s}")
print("-" * 80)

correct_pt, correct_onnx, match_count, total = 0, 0, 0, 0

for folder in sorted(DATA_DIR.iterdir()):
    if not folder.is_dir(): continue
    true_cls = None
    for cls in classes:
        if folder.name.endswith(cls):
            true_cls = cls; break
    if true_cls is None: continue

    imgs = list(folder.glob('*.jpg'))[:5]
    for img_path in imgs:
        img_pil = Image.open(img_path).convert('RGB')
        tensor  = transform(img_pil).unsqueeze(0)  # (1, 3, 224, 224)

        # PyTorch prediction
        with torch.no_grad():
            pt_out   = torch.softmax(pt_model(tensor), dim=1)
            pt_conf, pt_idx = pt_out.max(1)
        pt_pred = classes[pt_idx.item()]

        # ONNX prediction
        onnx_out  = sess.run(None, {'image': tensor.numpy()})[0]
        onnx_prob = np.exp(onnx_out) / np.exp(onnx_out).sum()  # softmax
        onnx_pred = classes[onnx_prob.argmax()]
        onnx_conf = onnx_prob.max()

        match = "✅" if pt_pred == onnx_pred else "❌"
        if pt_pred   == true_cls: correct_pt   += 1
        if onnx_pred == true_cls: correct_onnx += 1
        if pt_pred   == onnx_pred: match_count += 1
        total += 1

        print(f"{true_cls:20s} {pt_pred:20s} {onnx_pred:20s} {match:6s} {onnx_conf:.1%}")

print("-" * 80)
print(f"\nResults over {total} images:")
print(f"  PyTorch accuracy : {100*correct_pt/total:.1f}%")
print(f"  ONNX accuracy    : {100*correct_onnx/total:.1f}%")
print(f"  PT vs ONNX match : {100*match_count/total:.1f}%")

if match_count / total > 0.98:
    print("\n✅ ONNX model is correct — ready for Jetson deployment")
else:
    print("\n⚠️  PT and ONNX outputs differ — export may have failed")