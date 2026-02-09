import cv2
import torch

from models.mobilenetv2_heatmap import MobileNetV2Heatmap
from utils.heatmap_decode import heatmaps_to_keypoints
from filters.one_euro import OneEuroFilter
from geometry.hand_geometry import extract_geometry_features


device = "cuda" if torch.cuda.is_available() else "cpu"

model = MobileNetV2Heatmap(num_keypoints=21).to(device)
model.eval()

filter = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.4)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press q to quit")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray.astype("float32") / 255.0

        x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

        heatmaps = model(x)
        keypoints = heatmaps_to_keypoints(heatmaps)[0]
        keypoints = filter.filter(keypoints)

        features = extract_geometry_features(keypoints)

        y = 30
        for k, v in features.items():
            text = f"{k}: {v:.3f}" if torch.is_tensor(v) else f"{k}: {v}"
            cv2.putText(frame, text, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            y += 25

        cv2.imshow("Geometry Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
