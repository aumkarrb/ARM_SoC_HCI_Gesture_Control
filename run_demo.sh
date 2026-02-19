#!/usr/bin/env bash
# ==============================================================================
#  run_demo.sh  –  Preflight Checker & Demo Launcher
#  ARM SoC HCI Gesture Control – Jetson Nano
# ==============================================================================
#
#  USAGE
#  -----
#    ./run_demo.sh                          # auto-detects video in current dir
#    ./run_demo.sh --video /path/to/video   # specify video explicitly
#    ./run_demo.sh --camera 1               # use camera index 1 instead of 0
#    ./run_demo.sh --no-launch              # don't auto-launch VLC (manual mode)
#    ./run_demo.sh --skip-checks            # skip preflight, just run
#
#  WHAT THIS DOES
#  --------------
#    1. Validates all required files are present
#    2. Checks Python dependencies (onnxruntime, opencv, vlc_ipc)
#    3. Confirms a camera is reachable
#    4. Validates the ONNX model loads without error
#    5. Checks VLC is installed
#    6. Detects which ONNX execution provider will be used
#    7. Launches vlc_gesture_control.py with the right arguments
#    8. On exit: cleans up stray VLC / Python processes
#
# ==============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
CYN='\033[0;36m'
BLD='\033[1m'
RST='\033[0m'

pass() { echo -e "  ${GRN}[PASS]${RST}  $*"; }
fail() { echo -e "  ${RED}[FAIL]${RST}  $*"; FAILED=$((FAILED + 1)); }
warn() { echo -e "  ${YLW}[WARN]${RST}  $*"; }
info() { echo -e "  ${CYN}[INFO]${RST}  $*"; }

FAILED=0

# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

VIDEO_PATH=""
CAMERA_IDX="-1"        # -1 = auto-detect in vlc_gesture_control.py
AUTO_LAUNCH="true"
SKIP_CHECKS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)      VIDEO_PATH="$2";  shift 2 ;;
        --camera)     CAMERA_IDX="$2"; shift 2 ;;
        --no-launch)  AUTO_LAUNCH="false"; shift ;;
        --skip-checks) SKIP_CHECKS="true"; shift ;;
        --help|-h)
            sed -n '3,20p' "$0" | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${RST}"
            echo "Usage: $0 [--video PATH] [--camera N] [--no-launch] [--skip-checks]"
            exit 1
            ;;
    esac
done

# ── Resolve script directory (so the script works from any working dir) ───────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ══════════════════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLD}${CYN}╔══════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}${CYN}║   ARM SoC HCI Gesture Control – Demo Launcher       ║${RST}"
echo -e "${BLD}${CYN}║   Jetson Nano  │  EfficientNet-B0  │  VLC RC IPC    ║${RST}"
echo -e "${BLD}${CYN}╚══════════════════════════════════════════════════════╝${RST}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  SKIP CHECKS (fast path for repeated runs)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$SKIP_CHECKS" == "true" ]]; then
    warn "--skip-checks set: skipping preflight. Launching directly."
    LAUNCH_ARGS="--camera $CAMERA_IDX"
    [[ -n "$VIDEO_PATH" ]] && LAUNCH_ARGS="$LAUNCH_ARGS --video $VIDEO_PATH"
    python3.8 vlc_gesture_control.py $LAUNCH_ARGS
    exit $?
fi

# ══════════════════════════════════════════════════════════════════════════════
#  PREFLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

echo -e "${BLD}── 1 / 6  Required Files${RST}"
echo ""

# ── gesture_model.onnx ───────────────────────────────────────────────────────
if [[ -f "gesture_model.onnx" ]]; then
    SIZE=$(du -h gesture_model.onnx | cut -f1)
    pass "gesture_model.onnx  (${SIZE})"
else
    fail "gesture_model.onnx  NOT FOUND in $SCRIPT_DIR"
    echo -e "       ${YLW}Expected location: $SCRIPT_DIR/gesture_model.onnx${RST}"
fi

# ── classes.json ─────────────────────────────────────────────────────────────
if [[ -f "classes.json" ]]; then
    CLASSES=$(python3.8 -c "import json; c=json.load(open('classes.json')); print(', '.join(c))" 2>/dev/null || echo "parse error")
    pass "classes.json  →  [$CLASSES]"
else
    fail "classes.json  NOT FOUND in $SCRIPT_DIR"
fi

# ── vlc_gesture_control.py ───────────────────────────────────────────────────
if [[ -f "vlc_gesture_control.py" ]]; then
    pass "vlc_gesture_control.py"
else
    fail "vlc_gesture_control.py  NOT FOUND in $SCRIPT_DIR"
fi

# ── vlc_ipc.py ───────────────────────────────────────────────────────────────
if [[ -f "vlc_ipc.py" ]]; then
    pass "vlc_ipc.py"
else
    fail "vlc_ipc.py  NOT FOUND in $SCRIPT_DIR"
    warn "Download vlc_ipc.py and place it alongside vlc_gesture_control.py"
fi

# ── Video file (optional but warn if not provided) ───────────────────────────
if [[ -n "$VIDEO_PATH" ]]; then
    if [[ -f "$VIDEO_PATH" ]]; then
        VSIZE=$(du -h "$VIDEO_PATH" | cut -f1)
        pass "Video file: $VIDEO_PATH  (${VSIZE})"
    else
        fail "Video file NOT FOUND: $VIDEO_PATH"
    fi
else
    # Auto-detect first video file in current directory
    FOUND_VIDEO=$(find "$SCRIPT_DIR" -maxdepth 1 \
        \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.avi" \
           -o -iname "*.mov" -o -iname "*.webm" \) \
        -print -quit 2>/dev/null || true)
    if [[ -n "$FOUND_VIDEO" ]]; then
        VIDEO_PATH="$FOUND_VIDEO"
        VSIZE=$(du -h "$VIDEO_PATH" | cut -f1)
        pass "Video (auto-detected): $(basename "$VIDEO_PATH")  (${VSIZE})"
    else
        warn "No video file specified or found in $SCRIPT_DIR"
        warn "VLC will open without media. Use --video /path/to/file.mp4 to specify one."
    fi
fi

echo ""
echo -e "${BLD}── 2 / 6  Python Dependencies${RST}"
echo ""

# ── Python version ────────────────────────────────────────────────────────────
PY_VER=$(python3.8 --version 2>&1)
if python3.8 -c "import sys; assert sys.version_info >= (3,8)" 2>/dev/null; then
    pass "$PY_VER"
else
    fail "Python 3.8+ required. Found: $PY_VER"
fi

# ── onnxruntime ───────────────────────────────────────────────────────────────
if python3.8 -c "import onnxruntime" 2>/dev/null; then
    ORT_VER=$(python3.8 -c "import onnxruntime; print(onnxruntime.__version__)")
    pass "onnxruntime  $ORT_VER"
else
    fail "onnxruntime  NOT installed"
    echo -e "       ${YLW}Fix: pip3 install onnxruntime${RST}"
fi

# ── opencv ────────────────────────────────────────────────────────────────────
if python3.8 -c "import cv2" 2>/dev/null; then
    CV_VER=$(python3.8 -c "import cv2; print(cv2.__version__)")
    pass "opencv-python  $CV_VER"
else
    fail "opencv-python  NOT installed"
    echo -e "       ${YLW}Fix: pip3 install opencv-python${RST}"
fi

# ── numpy ─────────────────────────────────────────────────────────────────────
if python3.8 -c "import numpy" 2>/dev/null; then
    NP_VER=$(python3.8 -c "import numpy; print(numpy.__version__)")
    pass "numpy  $NP_VER"
else
    fail "numpy  NOT installed"
    echo -e "       ${YLW}Fix: pip3 install numpy${RST}"
fi

# ── vlc_ipc importable ────────────────────────────────────────────────────────
if python3.8 -c "import sys; sys.path.insert(0,'$SCRIPT_DIR'); import vlc_ipc" 2>/dev/null; then
    pass "vlc_ipc  (importable)"
else
    fail "vlc_ipc  import failed – check for syntax errors in vlc_ipc.py"
fi

# ── python-vlc (optional) ─────────────────────────────────────────────────────
if python3.8 -c "import vlc" 2>/dev/null; then
    pass "python-vlc  (optional – present)"
else
    warn "python-vlc  not installed (OK – using vlc_ipc socket method instead)"
fi

echo ""
echo -e "${BLD}── 3 / 6  Camera${RST}"
echo ""

# ── Camera detection ──────────────────────────────────────────────────────────
CAM_FOUND="false"
CAM_FOUND_IDX=""

if [[ "$CAMERA_IDX" != "-1" ]]; then
    # Specific index requested
    if python3.8 -c "
import cv2, sys
cap = cv2.VideoCapture($CAMERA_IDX)
if not cap.isOpened(): sys.exit(1)
ret, _ = cap.read()
cap.release()
sys.exit(0 if ret else 1)
" 2>/dev/null; then
        pass "Camera at index $CAMERA_IDX  is readable"
        CAM_FOUND="true"
        CAM_FOUND_IDX="$CAMERA_IDX"
    else
        fail "Camera at index $CAMERA_IDX  could not be opened"
    fi
else
    # Auto-detect camera
    for IDX in 0 1 2; do
        if python3.8 -c "
import cv2, sys
cap = cv2.VideoCapture($IDX)
if not cap.isOpened(): sys.exit(1)
ret, _ = cap.read()
cap.release()
sys.exit(0 if ret else 1)
" 2>/dev/null; then
            pass "Camera found at index $IDX"
            CAM_FOUND="true"
            CAM_FOUND_IDX="$IDX"
            CAMERA_IDX="$IDX"
            break
        fi
    done
    if [[ "$CAM_FOUND" == "false" ]]; then
        fail "No camera found at indices 0, 1, or 2"
        echo -e "       ${YLW}Fix: check USB webcam is connected. Try --camera N with your device index.${RST}"
    fi
fi

echo ""
echo -e "${BLD}── 4 / 6  ONNX Model Validation${RST}"
echo ""

# ── Load model and check input/output names + shape ───────────────────────────
if [[ -f "gesture_model.onnx" ]]; then
    python3.8 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(
        "gesture_model.onnx",
        providers=["CPUExecutionProvider"]   # always available for validation
    )

    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]

    print(f"  \033[0;36m[INFO]\033[0m  Input  name : {inp.name}")
    print(f"  \033[0;36m[INFO]\033[0m  Input  shape: {inp.shape}")
    print(f"  \033[0;36m[INFO]\033[0m  Output name : {out.name}")
    print(f"  \033[0;36m[INFO]\033[0m  Output shape: {out.shape}")

    # Run a dummy inference to confirm the model executes
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = sess.run([out.name], {inp.name: dummy})
    logits = result[0][0]

    print(f"  \033[0;32m[PASS]\033[0m  Dummy inference OK  →  logits shape: {logits.shape}")

    if logits.shape[0] != 6:
        print(f"  \033[0;33m[WARN]\033[0m  Expected 6 output classes, got {logits.shape[0]}")
    else:
        print(f"  \033[0;32m[PASS]\033[0m  Output classes: 6  ✓")

except Exception as e:
    print(f"  \033[0;31m[FAIL]\033[0m  ONNX model validation error: {e}")
    sys.exit(1)
PYEOF
    if [[ $? -ne 0 ]]; then
        FAILED=$((FAILED + 1))
    fi
else
    warn "Skipping model validation (gesture_model.onnx not found)"
fi

echo ""
echo -e "${BLD}── 5 / 6  VLC Installation${RST}"
echo ""

# ── VLC binary ────────────────────────────────────────────────────────────────
if command -v vlc &>/dev/null; then
    VLC_VER=$(vlc --version 2>&1 | head -1)
    pass "VLC found:  $VLC_VER"
else
    fail "VLC not found in PATH"
    echo -e "       ${YLW}Fix: sudo apt install vlc${RST}"
fi

echo ""
echo -e "${BLD}── 6 / 6  ONNX Execution Provider${RST}"
echo ""

# ── Detect best available provider ────────────────────────────────────────────
python3.8 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    import onnxruntime as ort
    available = ort.get_available_providers()
    priority  = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

    for p in priority:
        if p in available:
            best = p
            break
    else:
        best = available[0] if available else "None"

    labels = {
        "TensorrtExecutionProvider": ("TensorRT  (fastest – GPU + TRT)", "\033[0;32m[BEST]\033[0m "),
        "CUDAExecutionProvider":     ("CUDA      (fast  – GPU only)",     "\033[0;32m[PASS]\033[0m "),
        "CPUExecutionProvider":      ("CPU       (fallback – no GPU)",    "\033[0;33m[WARN]\033[0m "),
    }

    label, tag = labels.get(best, (best, "[INFO]"))
    print(f"  {tag}  Active provider: {label}")
    print(f"  \033[0;36m[INFO]\033[0m  All available:  {available}")

    if best == "CPUExecutionProvider":
        print(f"  \033[0;33m[WARN]\033[0m  Running on CPU – inference may be slow on Jetson.")
        print(f"         Install onnxruntime-gpu for CUDA/TRT acceleration.")
except Exception as e:
    print(f"  \033[0;31m[FAIL]\033[0m  Could not query ONNX providers: {e}")
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
#  PREFLIGHT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLD}─────────────────────────────────────────────────────${RST}"

if [[ $FAILED -gt 0 ]]; then
    echo -e "${BLD}${RED}  Preflight failed: $FAILED check(s) did not pass.${RST}"
    echo -e "${RED}  Fix the issues above before running the demo.${RST}"
    echo ""
    echo -e "  To bypass checks (not recommended): ${YLW}./run_demo.sh --skip-checks${RST}"
    echo ""
    exit 1
fi

echo -e "${BLD}${GRN}  All preflight checks passed. Ready to launch.${RST}"
echo -e "${BLD}─────────────────────────────────────────────────────${RST}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════════════════════

# Build the argument list for vlc_gesture_control.py
LAUNCH_ARGS="--camera $CAMERA_IDX"
[[ -n "$VIDEO_PATH" ]] && LAUNCH_ARGS="$LAUNCH_ARGS --video \"$VIDEO_PATH\""

echo -e "${BLD}  Launching gesture control …${RST}"
echo -e "  ${CYN}python3.8 vlc_gesture_control.py $LAUNCH_ARGS${RST}"
echo ""
echo -e "  ${YLW}Press Q in the camera window to quit.${RST}"
echo ""

# ── Cleanup function – runs on exit/Ctrl-C ────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${BLD}[EXIT]${RST}  Shutting down …"

    # Kill any VLC instances that were spawned by the gesture script
    # (uses the RC port as a fingerprint to avoid killing unrelated VLC windows)
    VLC_PIDS=$(pgrep -f "rc-host.*9595" 2>/dev/null || true)
    if [[ -n "$VLC_PIDS" ]]; then
        echo -e "  ${YLW}Stopping VLC (PIDs: $VLC_PIDS)${RST}"
        kill $VLC_PIDS 2>/dev/null || true
        sleep 0.5
    fi

    # Kill any lingering gesture control Python process
    GESTURE_PIDS=$(pgrep -f "vlc_gesture_control.py" 2>/dev/null || true)
    if [[ -n "$GESTURE_PIDS" ]]; then
        echo -e "  ${YLW}Stopping gesture control (PIDs: $GESTURE_PIDS)${RST}"
        kill $GESTURE_PIDS 2>/dev/null || true
    fi

    echo -e "${GRN}  Done. Goodbye.${RST}"
    echo ""
}

trap cleanup EXIT INT TERM

# ── Launch the gesture control script ─────────────────────────────────────────
eval python3.8 vlc_gesture_control.py $LAUNCH_ARGS

# eval returns the exit code of vlc_gesture_control.py
EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
    echo -e "${RED}  vlc_gesture_control.py exited with code $EXIT_CODE${RST}"
fi

exit $EXIT_CODE
