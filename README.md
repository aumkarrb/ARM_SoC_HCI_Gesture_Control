
 # ARM SoC Touchless HCI GESTURE-basd MEDIA CONTROL  on Jetson Nano (Maxwell, 4GB)
  

PROJECT GOAL
------------
Build a real-time hand gesture recognition system that runs on a Jetson Nano
to control media playback. Uses a webcam, classifies 6 hand gestures, and
triggers media actions.


FINAL RESULT
------------
  Model        : EfficientNet-B0 (pretrained ImageNet, fine-tuned on HaGRID)
  Dataset      : HaGRID-30k (30,000 images, 34,730 subjects, 6 gesture classes)
  Val Accuracy : 94.59%
  Test Accuracy: 94.60%
  ONNX Verified: 100% match between PyTorch and ONNX outputs
  Deployment   : ONNX Runtime (works on Jetson with or without TensorRT)


THE 6 GESTURE CLASSES
----------------------
  fist            → suggested action: pause / play
  ok              → suggested action: confirm / select
  palm            → suggested action: volume up
  stop            → suggested action: stop / exit
  two_up          → suggested action: next track
  two_up_inverted → suggested action: previous track

These map directly to HaGRID dataset classes. The model outputs one of these
6 class names with a confidence score (0-100%).


FILES TO DEPLOY ON JETSON
--------------------------
  gesture_model.onnx          — trained model (EfficientNet-B0)
  classes.json                — class label list
  jetson_inference_hagrid.py  — inference + camera script

Put all 3 in the same folder on Jetson. Nothing else needed.


JETSON SETUP (one time only)
-----------------------------
  pip3 install onnxruntime opencv-python

  Optional (faster inference with TensorRT):
  pip3 install onnxruntime-gpu

  Run:
  python3 jetson_inference_hagrid.py

  Press Q to quit.


HOW THE INFERENCE SCRIPT WORKS
--------------------------------
1. Opens camera (tries index 0, then 1)
2. Each frame: resize to 224x224, normalize with ImageNet stats
3. Pass through EfficientNet-B0 via ONNX Runtime
4. Softmax → class probabilities
5. Smooth predictions over last 5 frames (majority vote)
6. Only fire prediction if confidence > 70%
7. Display class name + confidence bar on screen
8. Print prediction to terminal

ONNX Runtime automatically uses the best available provider:
  TensorrtExecutionProvider  (fastest — use if TensorRT installed)
  CUDAExecutionProvider      (fast — use if CUDA available)
  CPUExecutionProvider       (fallback — always works)

No code changes needed regardless of which provider is used.
The script prints which provider is active at startup.


ADDING MEDIA CONTROL ACTIONS
------------------------------
To trigger actual media actions, add an action mapping to
jetson_inference_hagrid.py. After the prediction confidence check,
call your media control function.

Example structure (add inside the while loop after confidence check):

  if smooth_conf >= CONF_THRESHOLD:
      trigger_action(smooth_pred)

  def trigger_action(gesture):
      if gesture == 'fist':
          os.system('playerctl play-pause')    # Linux media control
      elif gesture == 'two_up':
          os.system('playerctl next')
      elif gesture == 'two_up_inverted':
          os.system('playerctl previous')
      elif gesture == 'palm':
          os.system('amixer sset Master 5%+')  # volume up
      elif gesture == 'stop':
          os.system('playerctl stop')

For playerctl (controls VLC, Spotify, etc. on Linux/Jetson):
  sudo apt install playerctl

For VLC specifically via python-vlc:
  pip3 install python-vlc

For keyboard simulation (works with any app):
  pip3 install pynput
  from pynput.keyboard import Key, Controller
  keyboard = Controller()
  keyboard.press(Key.media_next)   # next track


IMPORTANT: DEBOUNCE YOUR ACTIONS
----------------------------------
Add a cooldown so one gesture doesn't fire 30 times per second.
Example:

  import time
  last_action_time = 0
  COOLDOWN = 1.5  # seconds between actions

  if smooth_conf >= CONF_THRESHOLD:
      now = time.time()
      if now - last_action_time > COOLDOWN:
          trigger_action(smooth_pred)
          last_action_time = now


MODEL ARCHITECTURE (for reference)
------------------------------------
  Backbone  : EfficientNet-B0 (pretrained on ImageNet 1K)
              5.3M parameters
              Input: (batch, 3, 224, 224) float32
              ImageNet normalized: mean=[0.485,0.456,0.406]
                                   std =[0.229,0.224,0.225]

  Head      : Dropout(0.4) → Linear(1280, 6)

  Training  : Differential LR (backbone lr/10, head lr)
              AdamW + OneCycleLR
              Label smoothing 0.1
              Heavy augmentation (ColorJitter, RandomErasing, etc.)
              Weighted sampling for class balance

  ONNX      : Opset 11, dynamic batch axis
              Input name : 'image'
              Output name: 'logits'
              Softmax NOT included in ONNX — apply manually:
                probs = np.exp(logits) / np.exp(logits).sum()


PREPROCESSING (must match training exactly)
--------------------------------------------
  1. Read frame (BGR from OpenCV)
  2. Resize to 224x224
  3. Convert BGR → RGB
  4. Divide by 255.0 (float32)
  5. Subtract mean: [0.485, 0.456, 0.406]
  6. Divide by std:  [0.229, 0.224, 0.225]
  7. Transpose: (H, W, 3) → (3, H, W)
  8. Add batch dim: (3, H, W) → (1, 3, H, W)
  9. Pass to ONNX Runtime

In code:
  img = cv2.resize(frame, (224, 224))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
  inp = img.transpose(2,0,1)[np.newaxis]   # (1, 3, 224, 224)


WHAT DIDN'T WORK AND WHY (lessons learned)
--------------------------------------------
We tried several approaches before finding the working solution:

1. MediaPipe keypoints + CNN+LSTM
   Problem: Only 2 subjects in the IPN Hand dataset download.
   With 2 subjects, the model memorized appearance instead of gesture patterns.
   Accuracy ceiling: ~65% regardless of architecture or hyperparameters.
   Lesson: Subject diversity matters more than model architecture.

2. Raw frames + ResNet18 + BiLSTM on IPN Hand
   Problem: Same 2-subject dataset. Train 95%, val 58%.
   Classic overfitting from insufficient subject variety.

3. HaGRID dataset + EfficientNet-B0 (FINAL SOLUTION)
   34,730 subjects → genuine generalization
   Static gestures → no temporal model needed
   Result: 94.6% test accuracy, no overfitting, clean deployment


# QUICK REFERENCE: RUNNING ON JETSON

  ## One-time setup
  ```
  pip3 install onnxruntime opencv-python
```
  ## Run
 ```
 python3 jetson_inference_hagrid.py
```
  ## Terminal output while running:
  ###  fist                  87%
  ###   two_up                94%
  ###  palm                  91%

- Press Control+C to stop execution (for model + media control test)
- To start running, run:
```
python3.8 vlc_gesture_control.py --camera 0 \
  --video "/home/behura-san/jetson-test/ARM_SoC_HCI_Gesture_Control/test/video-a.mp4" \
  --video "/home/behura-san/jetson-test/ARM_SoC_HCI_Gesture_Control/test/video-b.mp4" \
  --video "/home/behura-san/jetson-test/ARM_SoC_HCI_Gesture_Control/test/video-c.mp4"
  ```



