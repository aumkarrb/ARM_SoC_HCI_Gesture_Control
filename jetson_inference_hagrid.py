"""
jetson_inference_hagrid.py — Real-time gesture recognition on Jetson Nano
==========================================================================
Uses ONNX Runtime with TensorRT execution provider for fast inference.
Falls back to CPU if TensorRT not available.

Copy to Jetson along with:
  - gesture_model.onnx
  - classes.json

Setup on Jetson:
  pip3 install onnxruntime-gpu  (or onnxruntime for CPU)
  python3 jetson_inference_hagrid.py

Controls:
  Q = quit
"""

import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque, Counter

try:
    import onnxruntime as ort
except ImportError:
    print("Install: pip3 install onnxruntime-gpu")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
ONNX_PATH      = SCRIPT_DIR / 'gesture_model.onnx'
CLASSES_PATH   = SCRIPT_DIR / 'classes.json'
CONF_THRESHOLD = 0.70
SMOOTH_FRAMES  = 5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLORS = {
    'fist':           (0,   100, 255),
    'ok':             (0,   255, 100),
    'palm':           (255, 100, 0  ),
    'stop':           (0,   0,   255),
    'two_up':         (255, 255, 0  ),
    'two_up_inverted':(255, 0,   255),
}


def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    # Load classes
    if CLASSES_PATH.exists():
        classes = json.load(open(CLASSES_PATH))
    else:
        classes = ['fist','ok','palm','stop','two_up','two_up_inverted']
    print(f"Classes: {classes}")

    # Load ONNX with best available provider
    providers = ['TensorrtExecutionProvider',
                 'CUDAExecutionProvider',
                 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
    active = sess.get_providers()[0]
    print(f"✅ ONNX loaded  |  Provider: {active}")

    # Open camera (try CSI first for Jetson, fall back to USB)
    cap = None
    for cam_id in [0, 1]:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            print(f"✅ Camera {cam_id} opened")
            break
    if not cap or not cap.isOpened():
        print("❌ No camera found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pred_history = deque(maxlen=SMOOTH_FRAMES)
    conf_history = deque(maxlen=SMOOTH_FRAMES)
    frame_count  = 0
    import time
    fps_time = time.time()

    print("🎥 Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        # Inference every frame
        inp    = preprocess(frame)
        logits = sess.run(None, {'image': inp})[0][0]
        probs  = softmax(logits)
        idx    = probs.argmax()
        conf   = probs[idx]

        pred_history.append(idx)
        conf_history.append(conf)

        smooth_idx  = Counter(pred_history).most_common(1)[0][0]
        smooth_pred = classes[smooth_idx]
        smooth_conf = float(np.mean(conf_history))
        color       = COLORS.get(smooth_pred, (255, 255, 255))

        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
        else:
            fps = 0

        # Draw
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)

        if smooth_conf >= CONF_THRESHOLD:
            cv2.putText(frame, smooth_pred.upper(),
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"{smooth_conf:.0%}",
                        (w-120, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No gesture",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (150, 150, 150), 2, cv2.LINE_AA)

        # Confidence bar
        bar_w = int((w-40) * smooth_conf)
        cv2.rectangle(frame, (20, 65), (20+bar_w, 75), color, -1)

        # Provider + FPS info
        provider_short = active.replace('ExecutionProvider','')
        cv2.putText(frame, f"{provider_short}",
                    (20, h-35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100,255,100), 1, cv2.LINE_AA)
        if fps > 0:
            cv2.putText(frame, f"{fps:.0f} FPS",
                        (20, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100,255,100), 1, cv2.LINE_AA)

        # Log to terminal
        if smooth_conf >= CONF_THRESHOLD:
            print(f"\r  {smooth_pred:20s}  {smooth_conf:.0%}", end='', flush=True)

        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done")


if __name__ == '__main__':
    main()