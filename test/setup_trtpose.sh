#!/bin/bash
# =============================================================================
# setup_trtpose.sh — Install trt_pose_hand + download model weights
# =============================================================================
# Run this on:
#   (A) Jetson Nano (JetPack 4.6+)  — for real-time inference
#   (B) Linux PC / WSL2 with CUDA   — for dataset preprocessing
#
# Usage:
#   chmod +x setup_trtpose.sh
#   ./setup_trtpose.sh
# =============================================================================

set -e  # exit on error

echo ""
echo "============================================================"
echo "  trt_pose_hand Setup Script"
echo "============================================================"
echo ""

# ── Detect environment ────────────────────────────────────────────────────────
IS_JETSON=false
if [[ -f /etc/nv_tegra_release ]]; then
    IS_JETSON=true
    echo "  Detected: Jetson device"
    cat /etc/nv_tegra_release | head -1
else
    echo "  Detected: Linux PC / WSL2"
fi
echo ""

# ── CUDA check ────────────────────────────────────────────────────────────────
echo "[1/6] Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "  ERROR: CUDA not found."
    if $IS_JETSON; then
        echo "  Install JetPack: https://developer.nvidia.com/embedded/jetpack"
    else
        echo "  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        echo "  WSL2 users: follow https://docs.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl"
    fi
    exit 1
fi
nvcc --version
echo "  ✅ CUDA found"

# ── Python packages ──────────────────────────────────────────────────────────
echo ""
echo "[2/6] Installing Python packages..."

# Core
pip install numpy opencv-python Pillow tqdm

# PyTorch (Jetson has its own wheel; PC installs from PyPI)
if $IS_JETSON; then
    echo "  [Jetson] PyTorch should already be installed via JetPack."
    echo "  If missing: https://forums.developer.nvidia.com/t/pytorch-for-jetson"
    python3 -c "import torch; print('  PyTorch:', torch.__version__)"
else
    echo "  Installing PyTorch CUDA 11.8 (Linux PC)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

echo "  ✅ PyTorch ready"

# ── torch2trt (needed for TRT conversion on Jetson) ───────────────────────────
echo ""
echo "[3/6] Installing torch2trt..."
if python3 -c "import torch2trt" 2>/dev/null; then
    echo "  torch2trt already installed ✅"
else
    cd /tmp
    if [[ -d torch2trt ]]; then rm -rf torch2trt; fi
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
    cd torch2trt
    python3 setup.py install --user
    cd -
    echo "  ✅ torch2trt installed"
fi

# ── trt_pose ──────────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing trt_pose..."
if python3 -c "import trt_pose" 2>/dev/null; then
    echo "  trt_pose already installed ✅"
else
    # Install dependencies
    pip install tqdm cython pycocotools

    cd /tmp
    if [[ -d trt_pose ]]; then rm -rf trt_pose; fi
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
    cd trt_pose
    python3 setup.py install --user
    cd -
    echo "  ✅ trt_pose installed"
fi

# ── Download trt_pose_hand model weights ──────────────────────────────────────
echo ""
echo "[5/6] Downloading trt_pose_hand model weights..."

WEIGHTS_DIR="$(pwd)"
TOPOLOGY_DIR="${WEIGHTS_DIR}/preprocess"
mkdir -p "$TOPOLOGY_DIR"

# ResNet18 model (faster — recommended for Jetson Nano 4GB)
RESNET_WEIGHTS="hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth"
RESNET_URL="https://drive.google.com/uc?id=1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo"

# DenseNet121 model (more accurate)
DENSENET_WEIGHTS="hand_pose_densenet121_baseline_att_224x224_B_epoch_149.pth"
DENSENET_URL="https://drive.google.com/uc?id=1ecs8MXiVpQKZM1gWzwPFnfWKPBaT7m7X"

# hand_pose.json topology
TOPOLOGY_URL="https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose_hand/master/preprocess/hand_pose.json"

echo "  Downloading hand_pose.json topology..."
if [[ ! -f "$TOPOLOGY_DIR/hand_pose.json" ]]; then
    curl -L "$TOPOLOGY_URL" -o "$TOPOLOGY_DIR/hand_pose.json"
    echo "  ✅ hand_pose.json saved to $TOPOLOGY_DIR/"
else
    echo "  hand_pose.json already exists ✅"
fi

echo ""
echo "  ⚠️  Model weights (300MB) must be downloaded manually:"
echo "  (Google Drive links require browser or gdown)"
echo ""
echo "  Option A — Use gdown (recommended):"
echo "    pip install gdown"
echo "    gdown '$RESNET_URL' -O $RESNET_WEIGHTS"
echo "    gdown '$DENSENET_URL' -O $DENSENET_WEIGHTS"
echo ""
echo "  Option B — Download manually:"
echo "    ResNet18  : https://github.com/NVIDIA-AI-IOT/trt_pose_hand"
echo "    DenseNet  : (same page, Assets section)"
echo "    Save to   : $(pwd)/"
echo ""

# Try gdown if available
if pip install gdown -q 2>/dev/null; then
    if [[ ! -f "$RESNET_WEIGHTS" ]]; then
        echo "  Trying gdown for ResNet18 weights..."
        gdown "$RESNET_URL" -O "$RESNET_WEIGHTS" 2>/dev/null || \
            echo "  ⚠️  gdown failed — please download manually (see above)"
    else
        echo "  ResNet18 weights already exist ✅"
    fi
fi

# ── Verify installation ────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."

python3 - << 'EOF'
import sys

checks = {
    'torch':     'import torch; print(f"  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")',
    'trt_pose':  'import trt_pose; print("  trt_pose ✅")',
    'torch2trt': 'import torch2trt; print("  torch2trt ✅")',
    'cv2':       'import cv2; print(f"  OpenCV {cv2.__version__} ✅")',
    'numpy':     'import numpy as np; print(f"  NumPy {np.__version__} ✅")',
}

failed = []
for name, code in checks.items():
    try:
        exec(code)
    except ImportError:
        print(f"  {name} ❌")
        failed.append(name)

if failed:
    print(f"\n  ⚠️  Missing: {failed}")
    sys.exit(1)
else:
    print("\n  All packages OK ✅")
EOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Ensure model weights are in current directory:"
echo "     hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth"
echo ""
echo "  2. Process IPD dataset (Linux/WSL2, GPU-accelerated):"
echo '     python3 process_ipd_trtpose.py \'
echo '         --ipd_dir "/path/to/IPD dataset" \'
echo '         --output_dir ipd_processed \'
echo '         --model resnet18'
echo ""
echo "  3. Copy ipd_processed/ to Windows for training:"
echo '     (or train directly on Linux/WSL2)'
echo ""
echo "  4. After training, copy gesture_model.onnx to Jetson Nano:"
echo '     scp gesture_model.onnx user@JETSON_IP:~/gesture/'
echo ""
echo "  5. Run real-time inference on Jetson:"
echo '     python3 jetson_inference_trtpose.py \'
echo '         --gesture_model gesture_model.onnx \'
echo '         --hand_model hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth \'
echo '         --media_control'
echo ""