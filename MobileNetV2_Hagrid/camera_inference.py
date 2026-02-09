import cv2
import torch

from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from utils.heatmap_decode import heatmaps_to_keypoints
from filters.one_euro import OneEuroFilter
from geometry.hand_geometry import extract_geometry_features
from fsm.gesture_fsm import GestureFSM
from actions.action_mapper import ActionMapper


# --------------------------------------------------
# Device
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Load model
# --------------------------------------------------
model = MobileNetV2Heatmap(num_keypoints=21)
model.to(device)
model.eval()


# --------------------------------------------------
# Filters, FSM, Actions
# --------------------------------------------------
keypoint_filter = OneEuroFilter(
    freq=30,          # camera FPS
    min_cutoff=1.0,
    beta=0.4,
    d_cutoff=1.0
)

fsm = GestureFSM(stable_frames=5)
action_mapper = ActionMapper(cooldown=1.0)


# --------------------------------------------------
# Camera
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open webcam")

print("Camera started. Press 'q' to quit.")


# --------------------------------------------------
# Main loop
# --------------------------------------------------
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------- Preprocess frame --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray.astype("float32") / 255.0

        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

        # -------- Model forward --------
        heatmaps = model(x)

        # -------- Decode keypoints --------
        keypoints = heatmaps_to_keypoints(heatmaps)[0]

        # -------- Temporal smoothing --------
        keypoints = keypoint_filter.filter(keypoints)

        # -------- Geometry + FSM --------
        features = extract_geometry_features(keypoints)
        gesture = fsm.update(features)

        # -------- Action mapping --------
        action_mapper.handle(gesture)

        # -------- Draw keypoints --------
        h, w, _ = frame.shape
        for (x_kp, y_kp) in keypoints:
            px = int(x_kp.item() * w)
            py = int(y_kp.item() * h)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        # -------- Display gesture --------
        cv2.putText(
            frame,
            f"Gesture: {gesture.name}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()
