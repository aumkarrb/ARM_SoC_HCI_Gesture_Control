"""
webcam_test.py — Live gesture recognition via laptop webcam
============================================================
Uses the exported ONNX model to classify gestures in real-time.
Press Q to quit.

Usage:
  python3 webcam_test.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    print("Run: pip install onnxruntime --break-system-packages")
    exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR       = Path('checkpoints_hagrid')
ONNX_PATH      = CKPT_DIR / 'gesture_model.onnx'
CONF_THRESHOLD = 0.70   # Only show prediction if confidence > 70%
SMOOTH_FRAMES  = 5      # Average predictions over last N frames

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Gesture → emoji for display
GESTURE_EMOJI = {
    'fist':           '✊ FIST',
    'ok':             '👌 OK',
    'palm':           '✋ PALM',
    'stop':           '🛑 STOP',
    'two_up':         '✌️  TWO UP',
    'two_up_inverted':'🤞 TWO UP INV',
}

# Colors per class (BGR)
COLORS = {
    'fist':           (0,   100, 255),
    'ok':             (0,   255, 100),
    'palm':           (255, 100, 0  ),
    'stop':           (0,   0,   255),
    'two_up':         (255, 255, 0  ),
    'two_up_inverted':(255, 0,   255),
}


def preprocess(frame):
    """Resize, normalize, convert to NCHW float32."""
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 224, 224)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    # Load model
    ckpt    = __import__('torch').load(
        str(CKPT_DIR / 'best_model.pth'), map_location='cpu', weights_only=False)
    classes = ckpt['classes']
    print(f"Classes: {classes}")

    sess = ort.InferenceSession(
        str(ONNX_PATH),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"✅ ONNX loaded  |  Provider: {sess.get_providers()[0]}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🎥 Webcam open — show a gesture to the camera")
    print("   Press Q to quit\n")

    pred_history = deque(maxlen=SMOOTH_FRAMES)
    conf_history = deque(maxlen=SMOOTH_FRAMES)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)   # Mirror

        # ── Inference ──────────────────────────────────────────────────────
        inp   = preprocess(frame)
        logits = sess.run(None, {'image': inp})[0][0]
        probs  = softmax(logits)
        idx    = probs.argmax()
        conf   = probs[idx]
        pred   = classes[idx]

        pred_history.append(idx)
        conf_history.append(conf)

        # Smoothed prediction (majority vote)
        from collections import Counter
        smooth_pred_idx  = Counter(pred_history).most_common(1)[0][0]
        smooth_pred      = classes[smooth_pred_idx]
        smooth_conf      = np.mean(conf_history)
        color            = COLORS.get(smooth_pred, (255, 255, 255))

        # ── Draw overlay ───────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Background bar at top
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)

        if smooth_conf >= CONF_THRESHOLD:
            label = GESTURE_EMOJI.get(smooth_pred, smooth_pred)
            cv2.putText(frame, label,
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        color, 3, cv2.LINE_AA)
            conf_text = f"{smooth_conf:.0%}"
            cv2.putText(frame, conf_text,
                        (w - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Show a gesture...",
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (150, 150, 150), 2, cv2.LINE_AA)

        # Confidence bar
        bar_w = int((w - 40) * smooth_conf)
        cv2.rectangle(frame, (20, 65), (20 + bar_w, 75), color, -1)
        cv2.rectangle(frame, (20, 65), (w - 20,     75), (80, 80, 80), 1)

        # All class probabilities on the right
        probs_smooth = softmax(logits)
        for i, (cls, prob) in enumerate(zip(classes, probs_smooth)):
            y    = 120 + i * 35
            bw   = int(200 * prob)
            col  = COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(frame, (w-220, y-18), (w-220+bw, y), col, -1)
            cv2.rectangle(frame, (w-220, y-18), (w-20,     y), (80,80,80), 1)
            cv2.putText(frame, f"{cls[:12]:12s} {prob:.0%}",
                        (w-215, y-3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255,255,255), 1, cv2.LINE_AA)

        # Instructions
        cv2.putText(frame, "Q = quit",
                    (20, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow('Gesture Recognition — 94.6% accuracy', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam closed")


if __name__ == '__main__':
    main()