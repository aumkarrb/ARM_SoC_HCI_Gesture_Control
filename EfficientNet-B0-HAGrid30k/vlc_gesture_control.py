import cv2, numpy as np, json, time, os, sys, argparse
from collections import Counter
from mpv_ipc import MPVSocketController as VLCController
try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[ERROR] onnxruntime not installed.")

MODEL_PATH      = "gesture_model.onnx"
CLASSES_PATH    = "classes.json"
CONF_THRESHOLD  = 0.70
SMOOTH_WINDOW   = 5
COOLDOWN_SEC    = 1.5
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GESTURE_ACTIONS = {
    "fist":            "Play / Pause",
    "ok":              "Next Chapter / +10s",
    "palm":            "Volume Up",
    "stop":            "Stop Playback",
    "two_up":          "Skip Forward +10s",
    "two_up_inverted": "Skip Backward -10s",
}

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, ...].astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=MODEL_PATH)
    parser.add_argument("--classes", default=CLASSES_PATH)
    parser.add_argument("--video", action="append", dest="videos",   default=None)
    parser.add_argument("--camera",  default=-1, type=int)
    args = parser.parse_args()

    with open(args.classes) as f:
        classes = json.load(f)
    print(f"[INFO] Loaded {len(classes)} gesture classes: {classes}")

    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    print(f"[INFO] ONNX Runtime provider: {sess.get_providers()[0]}")
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    vlc_ctrl = VLCController(playlist=args.videos)

    cap = None
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
    else:
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"[INFO] Camera opened at index {idx}")
                break
    if not cap or not cap.isOpened():
        sys.exit("[ERROR] Could not open any camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pred_window   = []
    last_action_t = 0

    print("\n[READY] Show a hand gesture to the camera. Press Ctrl+C to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        inp    = preprocess(frame)
        logits = sess.run([output_name], {input_name: inp})[0][0]
        probs  = softmax(logits)
        pred_idx   = int(np.argmax(probs))
        pred_conf  = float(probs[pred_idx])
        pred_class = classes[pred_idx]

        if pred_conf >= CONF_THRESHOLD:
            pred_window.append(pred_class)
        if len(pred_window) > SMOOTH_WINDOW:
            pred_window.pop(0)

        smooth_pred = None
        if pred_window:
            most_common, count = Counter(pred_window).most_common(1)[0]
            if count / len(pred_window) >= 0.6:
                smooth_pred = most_common

        now = time.time()
        if smooth_pred and (now - last_action_t) >= COOLDOWN_SEC:
            action = GESTURE_ACTIONS.get(smooth_pred, "")
            if action:
                vlc_ctrl.dispatch(smooth_pred)
                print(f"[ACTION] {smooth_pred}  ->  {action}")
                last_action_t = now

        time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C – shutting down.")
