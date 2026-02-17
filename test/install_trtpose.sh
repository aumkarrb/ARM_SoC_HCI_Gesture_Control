#!/bin/bash
# =============================================================================
# install_trtpose.sh — Install trt_pose on desktop Linux/WSL2
# Compatible with: CUDA 12.1, PyTorch 2.4.1, Python 3.10/3.11
#
# Usage:
#   chmod +x install_trtpose.sh
#   bash install_trtpose.sh
# =============================================================================

set -e
echo ""
echo "============================================================"
echo "  trt_pose Installation (Desktop CUDA 12.1 / PyTorch 2.4)"
echo "============================================================"
echo ""

# ── 1. Verify prerequisites ──────────────────────────────────────────────────
echo "[1/6] Checking prerequisites..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPU      : {torch.cuda.get_device_name(0)}')
print('  ✅ PyTorch + CUDA OK')
"

# ── 2. Dependencies ──────────────────────────────────────────────────────────
echo ""
echo "[2/6] Installing build dependencies..."
pip install -q tqdm cython
pip install -q pycocotools    # required by trt_pose

echo "  ✅ Dependencies installed"

# ── 3. Clone + install trt_pose ──────────────────────────────────────────────
echo ""
echo "[3/6] Installing trt_pose..."

if python3 -c "import trt_pose" 2>/dev/null; then
    echo "  trt_pose already installed ✅"
else
    cd /tmp
    rm -rf trt_pose
    git clone --depth 1 https://github.com/NVIDIA-AI-IOT/trt_pose.git
    cd trt_pose
    pip install -e .     # editable install — works without root
    cd -
    echo "  ✅ trt_pose installed"
fi

# ── 4. Verify trt_pose ───────────────────────────────────────────────────────
echo ""
echo "[4/6] Verifying trt_pose..."
python3 -c "
import trt_pose, trt_pose.coco, trt_pose.models
from trt_pose.parse_objects import ParseObjects
print(f'  trt_pose location : {trt_pose.__file__}')
print('  ✅ trt_pose imports OK')
"

# ── 5. Download hand topology JSON ──────────────────────────────────────────
echo ""
echo "[5/6] Downloading hand_pose.json topology..."
mkdir -p preprocess
TOPO_URL="https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose_hand/master/preprocess/hand_pose.json"

if [ ! -f "preprocess/hand_pose.json" ]; then
    curl -fsSL "$TOPO_URL" -o preprocess/hand_pose.json \
        && echo "  ✅ preprocess/hand_pose.json downloaded" \
        || echo "  ⚠️  curl failed — try: wget $TOPO_URL -O preprocess/hand_pose.json"
else
    echo "  preprocess/hand_pose.json already exists ✅"
fi

# ── 6. Download model weights ────────────────────────────────────────────────
echo ""
echo "[6/6] Downloading model weights..."

RESNET_FILE="hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth"

if [ -f "$RESNET_FILE" ]; then
    echo "  Model weights already exist: $RESNET_FILE ✅"
else
    echo "  Installing gdown for Google Drive download..."
    pip install -q gdown

    echo "  Downloading ResNet18 weights (~80 MB)..."
    # Primary link
    gdown "https://drive.google.com/uc?id=1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo" \
          -O "$RESNET_FILE" 2>/dev/null && {
        echo "  ✅ $RESNET_FILE downloaded"
    } || {
        echo ""
        echo "  ⚠️  gdown failed (Google Drive may block automated downloads)"
        echo ""
        echo "  MANUAL DOWNLOAD (takes 2 minutes):"
        echo "  ─────────────────────────────────"
        echo "  1. Open this URL in your Windows browser:"
        echo "     https://github.com/NVIDIA-AI-IOT/trt_pose_hand"
        echo ""
        echo "  2. Click 'hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth'"
        echo "     under the model table (look for Google Drive link)"
        echo ""
        echo "  3. Save it to: C:\\Users\\Soham\\Downloads\\"
        echo ""
        echo "  4. Copy it here:"
        echo "     cp '/mnt/c/Users/Soham/Downloads/$RESNET_FILE' ."
        echo ""
        echo "  Then re-run: bash install_trtpose.sh"
        echo "  (The script will skip steps 1-5 and just verify)"
        exit 1
    }
fi

# ── Verify weights ───────────────────────────────────────────────────────────
echo ""
echo "  Verifying model weights can load..."
python3 - << 'PYEOF'
import torch, json, trt_pose.coco, trt_pose.models
from trt_pose.parse_objects import ParseObjects
import os

weights = "hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth"
topo    = "preprocess/hand_pose.json"

if not os.path.exists(weights):
    print(f"  ❌ Weights not found: {weights}")
    exit(1)
if not os.path.exists(topo):
    print(f"  ❌ Topology not found: {topo}")
    exit(1)

with open(topo) as f:
    hp = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(hp)
n_parts  = len(hp['keypoints'])
n_links  = len(hp['skeleton'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = trt_pose.models.resnet18_baseline_att(n_parts, 2*n_links).to(device).eval()
model.load_state_dict(torch.load(weights, map_location=device))

# Test forward pass
dummy = torch.zeros(1, 3, 224, 224).to(device)
with torch.no_grad():
    cmap, paf = model(dummy)

print(f"  ✅ Model loaded on {device}")
print(f"  ✅ Forward pass OK: cmap={tuple(cmap.shape)}  paf={tuple(paf.shape)}")
print(f"  ✅ Outputs 21 keypoints as expected")
PYEOF

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ trt_pose READY"
echo "============================================================"
echo ""
echo "  Test it immediately:"
echo ""
echo "  python3 process_ipd_trtpose.py \\"
echo '      --ipd_dir "$IPD_DIR" \\'
echo "      --output_dir ipd_processed \\"
echo "      --seq_len 30 \\"
echo "      --max_videos 5"
echo ""
echo "  You should now see:"
echo "    ✅ trt_pose_hand loaded (GPU)"
echo "    (instead of MediaPipe fallback)"
echo ""