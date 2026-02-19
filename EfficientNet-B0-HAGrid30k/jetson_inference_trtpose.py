#!/usr/bin/env python3
"""
Real-Time Gesture Inference on Jetson Nano
==========================================
Pipeline:
  Camera → trt_pose_hand (TensorRT, GPU) → CNN+LSTM (TensorRT, GPU) → Media Control

Why this is optimal per research papers:
  - trt_pose_hand: GPU keypoint extraction (Paper 4: "low-energy real-time HCI")
  - CNN+LSTM: temporal gesture classification (Papers 1-3: "98.99% accuracy")
  - Both run on TensorRT → maximum Jetson Nano throughput

RUN THIS ON JETSON NANO ONLY
Requires:
  - JetPack 4.6+ (TensorRT 8+, CUDA 10.2+)
  - trt_pose installed: https://github.com/NVIDIA-AI-IOT/trt_pose_hand
  - torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
  - Model files: hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth
                 gesture_model.onnx (trained on your PC)

Usage:
  python3 jetson_inference_trtpose.py \
      --gesture_model  gesture_model.onnx \
      --hand_model     hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth \
      --camera_id      0
"""

import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque

# ── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# ── trt_pose_hand ─────────────────────────────────────────────────────────────
try:
    import trt_pose.coco
    import trt_pose.models
    from trt_pose.parse_objects import ParseObjects
    from torch2trt import TRTModule
    TRT_POSE_AVAILABLE = True
except ImportError:
    print("ERROR: trt_pose not found. Install from https://github.com/NVIDIA-AI-IOT/trt_pose_hand")
    sys.exit(1)

# ── ONNX Runtime (for gesture model before TRT conversion) ───────────────────
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ── Optional: media control ────────────────────────────────────────────────────
try:
    import subprocess
    MEDIA_CONTROL = True
except ImportError:
    MEDIA_CONTROL = False


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

GESTURE_TO_ACTION = {
    'click_one':   'play_pause',
    'click_two':   'stop',
    'throw_left':  'previous_track',
    'throw_right': 'next_track',
    'zoom_in':     'volume_up',
    'zoom_out':    'volume_down',
}

ACTION_COLORS = {
    'play_pause':     (0, 200,   0),
    'stop':           (0,   0, 200),
    'previous_track': (200, 0, 200),
    'next_track':     (200, 100, 0),
    'volume_up':      (0, 200, 200),
    'volume_down':    (200, 200, 0),
}

MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()[:, None, None]
STD  = torch.Tensor([0.229, 0.224, 0.225]).cuda()[:, None, None]


# ─────────────────────────────────────────────────────────────────────────────
#  trt_pose_hand wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TrtPoseHandModel:
    """
    Wraps trt_pose_hand inference.

    Tries to load TensorRT-converted model (fast) → falls back to PyTorch model.
    """
    INPUT_SIZE = 224

    def __init__(self, weights_path, topology_path, use_trt=True):
        # Load topology
        with open(topology_path, 'r') as f:
            hand_pose = json.load(f)
        self.topology  = trt_pose.coco.coco_category_to_topology(hand_pose)
        num_parts = len(hand_pose['keypoints'])
        num_links = len(hand_pose['skeleton'])

        # Try TRT model first (fast on Jetson)
        trt_path = weights_path.replace('.pth', '_trt.pth')
        if use_trt and os.path.exists(trt_path):
            print(f"  Loading TRT model: {trt_path}")
            self.model = TRTModule()
            self.model.load_state_dict(torch.load(trt_path))
            print("  ✅ TensorRT model loaded (maximum speed)")
        else:
            print(f"  Loading PyTorch model: {weights_path}")
            if 'resnet18' in weights_path:
                self.model = trt_pose.models.resnet18_baseline_att(
                    num_parts, 2 * num_links).cuda().eval()
            else:
                self.model = trt_pose.models.densenet121_baseline_att(
                    num_parts, 2 * num_links).cuda().eval()
            self.model.load_state_dict(torch.load(weights_path))
            print("  ✅ PyTorch model loaded")

            # Auto-convert to TRT for future runs
            if use_trt:
                self._convert_to_trt(trt_path, num_parts, num_links)

        self.parse_objects = ParseObjects(
            self.topology, cmap_threshold=0.15, link_threshold=0.15
        )

    def _convert_to_trt(self, trt_save_path, num_parts, num_links):
        """Convert PyTorch model to TensorRT for faster inference."""
        try:
            from torch2trt import torch2trt
            print("  Converting to TensorRT (one-time, ~2 min)...")
            dummy = torch.ones(1, 3, self.INPUT_SIZE, self.INPUT_SIZE).cuda()
            model_trt = torch2trt(self.model, [dummy], fp16_mode=True)
            torch.save(model_trt.state_dict(), trt_save_path)
            self.model = model_trt
            print(f"  ✅ TRT model saved → {trt_save_path}")
        except Exception as e:
            print(f"  ⚠️  TRT conversion failed: {e}  (using PyTorch)")

    @torch.no_grad()
    def infer(self, bgr_frame):
        """
        Extract 21 hand keypoints from BGR frame.

        Returns: np.ndarray (42,) or None
        """
        img = cv2.resize(bgr_frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img).float().cuda() / 255.0
        t   = (t.permute(2, 0, 1) - MEAN) / STD
        t   = t.unsqueeze(0)

        cmap, paf = self.model(t)
        counts, objects, peaks = self.parse_objects(
            cmap.detach().cpu(), paf.detach().cpu()
        )

        if counts[0] == 0:
            return None

        obj = objects[0][0]
        kp  = np.zeros((21, 2), dtype=np.float32)
        for i in range(21):
            k = int(obj[i])
            if k >= 0:
                kp[i] = [peaks[0][i][k][1].item(),   # x
                          peaks[0][i][k][0].item()]   # y
        return kp.flatten()


# ─────────────────────────────────────────────────────────────────────────────
#  Gesture classifier (CNN+LSTM via ONNX or TRT)
# ─────────────────────────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Loads trained CNN+LSTM model and classifies gesture sequences.
    Supports ONNX (easy) and TensorRT (fastest on Jetson).
    """
    def __init__(self, model_path, class_mapping_path=None, seq_len=30, conf_threshold=0.7):
        self.seq_len        = seq_len
        self.conf_threshold = conf_threshold

        # Load class names
        if class_mapping_path and os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                d = json.load(f)
            self.classes = d['classes']
        else:
            self.classes = ['click_one', 'click_two', 'throw_left',
                            'throw_right', 'zoom_in', 'zoom_out']

        print(f"  Gesture classes: {self.classes}")

        # Load model
        if model_path.endswith('.onnx'):
            self._load_onnx(model_path)
        elif model_path.endswith('.pth'):
            self._load_pytorch(model_path)
        else:
            raise ValueError(f"Unknown model format: {model_path}")

    def _load_onnx(self, path):
        assert ONNX_AVAILABLE, "pip install onnxruntime"
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.mode = 'onnx'
        print(f"  ✅ ONNX model loaded: {path}")

    def _load_pytorch(self, path):
        import sys; sys.path.append(str(Path(__file__).parent))
        from models import build_model
        ckpt = torch.load(path, map_location='cuda')
        self.classes = ckpt.get('classes', self.classes)
        self.model = build_model('cnn_lstm',
                                 num_classes=len(self.classes),
                                 seq_len=ckpt.get('seq_len', self.seq_len))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model = self.model.cuda().eval()
        self.mode = 'pytorch'
        print(f"  ✅ PyTorch CNN+LSTM loaded: {path}")

    def normalize(self, kp_flat):
        """Same normalization as training."""
        kp = kp_flat.reshape(21, 2)
        kp -= kp[0]                                        # center wrist
        scale = np.linalg.norm(kp[12]) + 1e-6
        return (kp / scale).flatten()

    def predict(self, frame_buffer):
        """
        Predict gesture from buffered keypoint frames.

        Args:
            frame_buffer: list of (42,) arrays, length = seq_len
        Returns:
            (gesture_name, confidence) or (None, 0.0)
        """
        if len(frame_buffer) < self.seq_len:
            return None, 0.0

        seq = np.array([self.normalize(f) for f in frame_buffer],
                        dtype=np.float32)                  # (seq_len, 42)

        # Per-sequence normalization (same as GestureDataset)
        seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)

        if self.mode == 'onnx':
            inp = seq[np.newaxis, ...]                     # (1, 30, 42)
            logits = self.session.run([self.output_name], {self.input_name: inp})[0]
        else:
            with torch.no_grad():
                t = torch.from_numpy(seq[np.newaxis]).cuda()
                logits = self.model(t).cpu().numpy()

        probs = self._softmax(logits[0])
        idx   = np.argmax(probs)
        conf  = float(probs[idx])

        if conf >= self.conf_threshold:
            return self.classes[idx], conf
        return None, conf

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
#  Media control executor
# ─────────────────────────────────────────────────────────────────────────────

def execute_media_action(action):
    """Execute media control commands on Jetson Nano (Linux)."""
    commands = {
        'play_pause':     ['playerctl', 'play-pause'],
        'stop':           ['playerctl', 'stop'],
        'next_track':     ['playerctl', 'next'],
        'previous_track': ['playerctl', 'previous'],
        'volume_up':      ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '+10%'],
        'volume_down':    ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', '-10%'],
    }
    cmd = commands.get(action)
    if cmd:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"  Media control error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Draw overlay
# ─────────────────────────────────────────────────────────────────────────────

HAND_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

def draw_hand_overlay(frame, keypoints_flat, gesture=None, conf=0.0, fps=0.0):
    h, w = frame.shape[:2]
    kp = keypoints_flat.reshape(21, 2)

    # Draw skeleton
    for i, j in HAND_SKELETON:
        if kp[i].any() and kp[j].any():
            x1, y1 = int(kp[i][0] * w), int(kp[i][1] * h)
            x2, y2 = int(kp[j][0] * w), int(kp[j][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (100, 200, 100), 2)

    # Draw keypoints
    for i, (x_n, y_n) in enumerate(kp):
        if x_n > 0 or y_n > 0:
            x, y = int(x_n * w), int(y_n * h)
            color = (0, 0, 255) if i == 0 else (255, 100, 0)
            cv2.circle(frame, (x, y), 4, color, -1)

    # HUD
    cv2.rectangle(frame, (0, 0), (380, 100), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if gesture:
        action  = GESTURE_TO_ACTION.get(gesture, '')
        color   = ACTION_COLORS.get(action, (0, 255, 0))
        cv2.putText(frame, f"{gesture}  ({conf:.0%})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"→ {action}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args):
    print("\n" + "="*60)
    print("  CNN+LSTM Gesture Inference — trt_pose_hand pipeline")
    print("="*60)

    # ── Find topology ──────────────────────────────────────────────────
    topology_candidates = [
        'preprocess/hand_pose.json',
        'hand_pose.json',
        os.path.join(os.path.expanduser('~'), 'trt_pose/tasks/hand_pose/preprocess/hand_pose.json'),
    ]
    topology_path = None
    for c in topology_candidates:
        if os.path.exists(c):
            topology_path = c
            break
    if topology_path is None:
        print("ERROR: hand_pose.json not found. Run: bash setup_trtpose.sh")
        sys.exit(1)

    # ── Load trt_pose_hand ──────────────────────────────────────────────
    print("\n[1/3] Loading trt_pose_hand keypoint extractor...")
    hand_model = TrtPoseHandModel(
        weights_path=args.hand_model,
        topology_path=topology_path,
        use_trt=not args.no_trt
    )

    # ── Load CNN+LSTM gesture classifier ───────────────────────────────
    print("\n[2/3] Loading CNN+LSTM gesture classifier...")
    mapping_path = args.gesture_model.replace('.onnx', '.json').replace('.pth', '.json')
    classifier = GestureClassifier(
        model_path=args.gesture_model,
        class_mapping_path=mapping_path,
        seq_len=args.seq_len,
        conf_threshold=args.conf_threshold
    )

    # ── Open camera ────────────────────────────────────────────────────
    print(f"\n[3/3] Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_id}")
        sys.exit(1)

    print("\n✅ Pipeline ready! Press 'q' to quit.\n")
    print(f"  Gesture buffer  : {args.seq_len} frames")
    print(f"  Conf threshold  : {args.conf_threshold:.0%}")
    print(f"  Media control   : {'ON' if args.media_control else 'OFF'}")

    # ── Inference loop ─────────────────────────────────────────────────
    frame_buffer     = deque(maxlen=args.seq_len)
    fps_deque        = deque(maxlen=30)
    current_gesture  = None
    current_conf     = 0.0
    last_action_time = 0.0
    ACTION_COOLDOWN  = 1.5                  # seconds between actions
    last_kp          = None

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)         # mirror for natural feel

        # trt_pose_hand → keypoints
        kp = hand_model.infer(frame)
        if kp is not None:
            last_kp = kp.copy()
            frame_buffer.append(kp)

        # CNN+LSTM → gesture (runs every frame when buffer is full)
        if len(frame_buffer) == args.seq_len:
            gesture, conf = classifier.predict(list(frame_buffer))
            if gesture is not None:
                current_gesture = gesture
                current_conf    = conf

                # Trigger media action with cooldown
                now = time.time()
                if args.media_control and (now - last_action_time) > ACTION_COOLDOWN:
                    action = GESTURE_TO_ACTION.get(gesture)
                    if action:
                        execute_media_action(action)
                        last_action_time = now
                        print(f"  ▶  {gesture} → {action}  ({conf:.0%})")

        # FPS calculation
        fps_deque.append(time.time() - t0)
        fps = 1.0 / (sum(fps_deque) / len(fps_deque) + 1e-9)

        # Draw
        if last_kp is not None:
            frame = draw_hand_overlay(frame, last_kp, current_gesture, current_conf, fps)
        else:
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('CNN+LSTM Gesture Control — trt_pose_hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CNN+LSTM Gesture Inference with trt_pose_hand')
    p.add_argument('--gesture_model',   type=str, required=True,
                   help='Path to trained model (.onnx or .pth)')
    p.add_argument('--hand_model',      type=str,
                   default='hand_pose_resnet18_baseline_att_224x224_A_epoch_249.pth',
                   help='trt_pose_hand weights file')
    p.add_argument('--camera_id',       type=int, default=0)
    p.add_argument('--seq_len',         type=int, default=30,
                   help='Frames to buffer before classifying')
    p.add_argument('--conf_threshold',  type=float, default=0.70,
                   help='Minimum confidence to trigger action')
    p.add_argument('--media_control',   action='store_true',
                   help='Enable actual media commands (playerctl/pactl)')
    p.add_argument('--no_trt',          action='store_true',
                   help='Disable TensorRT conversion (use PyTorch)')
    args = p.parse_args()

    run_inference(args)