"""
gesture_guide.py  –  Audience-Facing Live Reference Display
============================================================
ARM SoC HCI Gesture Control – Jetson Nano

PURPOSE
-------
A second OpenCV window, intended for the AUDIENCE or a second monitor /
projector, that shows:
  • All 6 gesture cards with their labels and mapped VLC actions
  • A live "LAST DETECTED" highlight on whichever gesture was just fired
  • A real-time confidence bar
  • A "LAST ACTION" feedback banner (e.g. "Paused", "+10s")
  • A cooldown indicator so the audience knows when the next gesture is ready

This window reads a tiny shared-state JSON file written by
vlc_gesture_control.py every frame.  It is completely decoupled:
  • It cannot crash the inference engine
  • It can be started / stopped independently at any time
  • It runs in its own process – no threads shared with the main script

HOW TO INTEGRATE
----------------
1.  Add the state-writer snippet to vlc_gesture_control.py
    (exact lines are printed when you run this file with --print-patch).

2.  Start the guide in a second terminal BEFORE or AFTER the main script:
        python3 gesture_guide.py

3.  Move the guide window to your projector / second monitor.
    The camera+HUD window stays on the presenter's screen.

USAGE
-----
    python3 gesture_guide.py                        # default state file path
    python3 gesture_guide.py --state /tmp/gs.json   # custom path
    python3 gesture_guide.py --fullscreen            # fullscreen mode
    python3 gesture_guide.py --print-patch           # print the vlc_gesture_control.py patch

DEPENDENCIES
------------
    pip3 install opencv-python numpy      (already required by main script)

Press Q in the guide window to close it.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

STATE_FILE      = "/tmp/gesture_state.json"   # must match PATCH below
POLL_INTERVAL   = 0.033                        # ~30 fps refresh
STALE_TIMEOUT   = 3.0    # seconds before marking inference as "offline"
ACTION_HOLD     = 2.5    # seconds to highlight last action banner
COOLDOWN_SEC    = 1.5    # mirrored from vlc_gesture_control.py for indicator

# Window dimensions  (landscape, good for 1080p projector)
WIN_W, WIN_H = 1280, 720

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
C = {
    "bg":           (18,  18,  28),    # near-black blue background
    "card_idle":    (35,  35,  52),    # inactive gesture card
    "card_active":  (0,   140, 60),    # active gesture card (green)
    "card_border":  (60,  60,  90),    # card border default
    "card_hi_bdr":  (0,   220, 100),   # card border when active
    "title":        (200, 200, 255),   # header text
    "label":        (255, 255, 255),   # gesture name text
    "action_txt":   (180, 220, 255),   # action description text
    "conf_bg":      (50,  50,  70),    # confidence bar background
    "conf_fill":    (0,   200, 80),    # confidence bar fill (green)
    "conf_cd":      (60,  60, 200),    # confidence bar fill during cooldown (blue)
    "banner_bg":    (0,   100, 200),   # last-action banner
    "banner_txt":   (255, 255, 255),
    "offline":      (80,  30,  30),    # offline / waiting state
    "offline_txt":  (160, 100, 100),
    "white":        (255, 255, 255),
    "grey":         (130, 130, 130),
}

# ── Gesture metadata ──────────────────────────────────────────────────────────
GESTURES = [
    {
        "name":        "fist",
        "label":       "FIST",
        "symbol":      "F",            # ASCII fallback (OpenCV can't render emoji)
        "action":      "Play / Pause",
        "hint":        "Close all fingers tightly",
    },
    {
        "name":        "ok",
        "label":       "OK",
        "symbol":      "O",
        "action":      "Next Chapter",
        "hint":        "Thumb and index finger circle",
    },
    {
        "name":        "palm",
        "label":       "PALM",
        "symbol":      "P",
        "action":      "Volume Up",
        "hint":        "Open hand, fingers spread",
    },
    {
        "name":        "stop",
        "label":       "STOP",
        "symbol":      "S",
        "action":      "Stop Playback",
        "hint":        "Flat hand, palm facing camera",
    },
    {
        "name":        "two_up",
        "label":       "TWO UP",
        "symbol":      "2",
        "action":      "Skip +10s",
        "hint":        "Index + middle fingers up",
    },
    {
        "name":        "two_up_inverted",
        "label":       "TWO DOWN",
        "symbol":      "2v",
        "action":      "Skip -10s",
        "hint":        "Index + middle fingers down",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  STATE FILE PATCH  (printed with --print-patch)
# ══════════════════════════════════════════════════════════════════════════════

PATCH_TEXT = '''
# ── Add these imports near the top of vlc_gesture_control.py ──────────────────
import json, tempfile, os

# ── Add this constant near the other constants (after COOLDOWN_SEC) ──────────
STATE_FILE = "/tmp/gesture_state.json"

# ── Add this function anywhere before main() ─────────────────────────────────
def write_state(gesture, confidence, action, in_cooldown):
    """Write current inference state for gesture_guide.py to read."""
    state = {
        "gesture":     gesture,
        "confidence":  round(float(confidence), 4),
        "action":      action,
        "in_cooldown": in_cooldown,
        "timestamp":   time.time(),
    }
    # Atomic write: write to temp file then rename to avoid partial reads
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass   # non-fatal – guide display just shows stale data

# ── Inside the while loop in main(), REPLACE the "Draw HUD" block with ───────
# ── (add the write_state call right before draw_hud) ─────────────────────────

        # Write state for gesture_guide.py
        write_state(
            gesture     = display_gesture,
            confidence  = display_conf,
            action      = last_action_msg,
            in_cooldown = in_cooldown,
        )

        # Draw HUD  (this line already exists – keep it as-is)
        frame = draw_hud(frame, display_gesture, display_conf,
                         last_action_msg, in_cooldown, classes)
'''


# ══════════════════════════════════════════════════════════════════════════════
#  STATE READER
# ══════════════════════════════════════════════════════════════════════════════

def read_state(state_file: str) -> dict:
    """
    Read the latest inference state from the JSON file.
    Returns a default 'waiting' state if the file is missing, stale, or corrupt.
    """
    default = {
        "gesture":     None,
        "confidence":  0.0,
        "action":      "",
        "in_cooldown": False,
        "timestamp":   0.0,
        "online":      False,
    }
    try:
        with open(state_file, "r") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        data["online"] = age < STALE_TIMEOUT
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rounded_rect(img, x1, y1, x2, y2, radius, colour, thickness=-1):
    """Draw a rectangle with rounded corners."""
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), colour, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), colour, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, colour, -1)
    else:
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, colour, thickness)
        cv2.line(img, (x1+r, y1), (x2-r, y1), colour, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), colour, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), colour, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), colour, thickness)


def put_text_centered(img, text, cx, cy, font, scale, colour, thickness=1):
    """Draw text centred on (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), font, scale, colour, thickness,
                cv2.LINE_AA)


def draw_confidence_bar(img, x, y, w, h, confidence, in_cooldown):
    """Draw a horizontal confidence bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C["conf_bg"], -1)
    fill_w = int(w * max(0.0, min(1.0, confidence)))
    fill_col = C["conf_cd"] if in_cooldown else C["conf_fill"]
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_col, -1)
    # Border
    cv2.rectangle(img, (x, y), (x + w, y + h), C["conf_bg"], 1)
    # Percentage label
    pct_text = f"{int(confidence * 100)}%"
    cv2.putText(img, pct_text, (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["white"], 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_frame(state: dict, last_action_time: float) -> np.ndarray:
    """
    Render the full guide frame from the current state dict.
    Returns a (WIN_H, WIN_W, 3) uint8 BGR array.
    """
    img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    img[:] = C["bg"]

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_bold   = cv2.FONT_HERSHEY_DUPLEX
    active_name = state.get("gesture")
    confidence  = state.get("confidence", 0.0)
    action_msg  = state.get("action", "")
    in_cooldown = state.get("in_cooldown", False)
    online      = state.get("online", False)

    # ── Header ────────────────────────────────────────────────────────────────
    header_h = 80
    cv2.rectangle(img, (0, 0), (WIN_W, header_h), (25, 25, 40), -1)
    cv2.putText(img, "ARM SoC HCI  –  Gesture Control Demo",
                (30, 30), font_bold, 0.75, C["title"], 1, cv2.LINE_AA)
    cv2.putText(img, "EfficientNet-B0  |  HaGRID-30k  |  6 Gesture Classes  |  VLC RC IPC",
                (30, 58), font, 0.42, C["grey"], 1, cv2.LINE_AA)

    # Online / Offline indicator (top right)
    status_txt   = "LIVE" if online else "WAITING FOR INFERENCE..."
    status_col   = (0, 200, 80) if online else C["offline_txt"]
    status_x     = WIN_W - 220
    cv2.circle(img, (status_x, 38), 8, status_col, -1)
    cv2.putText(img, status_txt, (status_x + 18, 44),
                font, 0.5, status_col, 1, cv2.LINE_AA)

    # Separator line
    cv2.line(img, (0, header_h), (WIN_W, header_h), (50, 50, 80), 1)

    # ── Gesture cards ─────────────────────────────────────────────────────────
    #   Layout: 3 columns × 2 rows, with padding
    n_cols      = 3
    n_rows      = 2
    pad_x       = 28
    pad_y       = 18
    cards_top   = header_h + 16
    cards_bot   = WIN_H - 130     # leave room for bottom panel
    card_w      = (WIN_W - pad_x * (n_cols + 1)) // n_cols
    card_h      = (cards_bot - cards_top - pad_y * (n_rows + 1)) // n_rows

    for idx, g in enumerate(GESTURES):
        col = idx % n_cols
        row = idx // n_cols

        x1 = pad_x + col * (card_w + pad_x)
        y1 = cards_top + pad_y + row * (card_h + pad_y)
        x2 = x1 + card_w
        y2 = y1 + card_h

        is_active = (g["name"] == active_name) and online

        # Card background
        bg_col  = C["card_active"] if is_active else C["card_idle"]
        bdr_col = C["card_hi_bdr"] if is_active else C["card_border"]
        bdr_w   = 3 if is_active else 1

        rounded_rect(img, x1, y1, x2, y2, 12, bg_col, -1)
        rounded_rect(img, x1, y1, x2, y2, 12, bdr_col, bdr_w)

        # Pulse effect: brighten border when just activated
        if is_active:
            pulse = abs(math.sin(time.time() * 4.0))
            bright = tuple(min(255, int(c + pulse * 80)) for c in bdr_col)
            rounded_rect(img, x1+1, y1+1, x2-1, y2-1, 11, bright, 2)

        # Symbol (big letter in centre-left of card)
        sym_cx = x1 + 52
        sym_cy = y1 + card_h // 2 - 8
        sym_col = (255, 255, 255) if is_active else (120, 120, 160)
        sym_bg  = (0, 100, 50) if is_active else (50, 50, 75)
        cv2.circle(img, (sym_cx, sym_cy), 34, sym_bg, -1)
        put_text_centered(img, g["symbol"], sym_cx, sym_cy,
                          font_bold, 0.85, sym_col, 2)

        # Gesture label
        lbl_col = (255, 255, 255) if is_active else C["label"]
        cv2.putText(img, g["label"],
                    (x1 + 100, y1 + 34),
                    font_bold, 0.72, lbl_col, 1, cv2.LINE_AA)

        # Action line
        act_col = (200, 255, 220) if is_active else C["action_txt"]
        cv2.putText(img, g["action"],
                    (x1 + 100, y1 + 62),
                    font, 0.52, act_col, 1, cv2.LINE_AA)

        # Hint line
        hint_col = (180, 255, 200) if is_active else (90, 90, 120)
        cv2.putText(img, g["hint"],
                    (x1 + 100, y1 + 85),
                    font, 0.38, hint_col, 1, cv2.LINE_AA)

        # Active glow line at bottom of card
        if is_active:
            cv2.line(img, (x1 + 14, y2 - 6), (x2 - 14, y2 - 6),
                     C["card_hi_bdr"], 3)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    panel_y = cards_bot + 10
    cv2.line(img, (0, panel_y - 10), (WIN_W, panel_y - 10), (50, 50, 80), 1)

    # Left: "DETECTED" label + gesture name
    detected_label = active_name.upper().replace("_", " ") if active_name else "---"
    cv2.putText(img, "DETECTED:", (30, panel_y + 22),
                font, 0.5, C["grey"], 1, cv2.LINE_AA)
    det_col = (0, 255, 140) if (online and active_name) else C["grey"]
    cv2.putText(img, detected_label, (30, panel_y + 52),
                font_bold, 1.0, det_col, 2, cv2.LINE_AA)

    # Centre: Confidence bar
    bar_x = 320
    bar_y = panel_y + 14
    bar_w = 380
    bar_h = 22
    cv2.putText(img, "CONFIDENCE", (bar_x, bar_y - 4),
                font, 0.42, C["grey"], 1, cv2.LINE_AA)
    draw_confidence_bar(img, bar_x, bar_y, bar_w, bar_h,
                        confidence if online else 0.0, in_cooldown)

    # Cooldown label
    if in_cooldown and online:
        cv2.putText(img, "COOLDOWN", (bar_x, bar_y + bar_h + 18),
                    font, 0.42, (130, 130, 210), 1, cv2.LINE_AA)
    elif online:
        cv2.putText(img, "READY", (bar_x, bar_y + bar_h + 18),
                    font, 0.42, (80, 200, 80), 1, cv2.LINE_AA)

    # Right: Last action banner
    now = time.time()
    action_age = now - last_action_time
    if action_msg and action_age < ACTION_HOLD:
        alpha = max(0.0, 1.0 - (action_age / ACTION_HOLD))   # fade out
        bx1, by1, bx2, by2 = WIN_W - 320, panel_y + 4, WIN_W - 24, panel_y + 68
        banner_col = tuple(int(c * alpha) for c in C["banner_bg"])
        rounded_rect(img, bx1, by1, bx2, by2, 10, banner_col, -1)
        cv2.putText(img, "LAST ACTION", (bx1 + 14, by1 + 20),
                    font, 0.4, tuple(int(c * alpha) for c in C["grey"]),
                    1, cv2.LINE_AA)
        txt_col = tuple(int(c * alpha) for c in C["banner_txt"])
        cv2.putText(img, action_msg, (bx1 + 14, by1 + 50),
                    font_bold, 0.75, txt_col, 1, cv2.LINE_AA)

    # Footer
    cv2.putText(img,
                "Q : close guide     |     run_demo.sh to launch full system",
                (30, WIN_H - 12),
                font, 0.38, (60, 60, 90), 1, cv2.LINE_AA)

    # Offline overlay
    if not online:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, header_h), (WIN_W, WIN_H), C["offline"], -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        put_text_centered(img,
                          "Waiting for vlc_gesture_control.py ...",
                          WIN_W // 2, WIN_H // 2,
                          font_bold, 0.9, C["offline_txt"], 1)
        put_text_centered(img,
                          f"Watching: {STATE_FILE}",
                          WIN_W // 2, WIN_H // 2 + 40,
                          font, 0.45, (80, 60, 60), 1)

    return img


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global STATE_FILE

    parser = argparse.ArgumentParser(
        description="gesture_guide.py – Audience display for gesture control demo"
    )
    parser.add_argument("--state",        default=STATE_FILE,
                        help=f"Path to shared state JSON (default: {STATE_FILE})")
    parser.add_argument("--fullscreen",   action="store_true",
                        help="Launch in fullscreen mode")
    parser.add_argument("--print-patch",  action="store_true",
                        help="Print the code to add to vlc_gesture_control.py and exit")
    args = parser.parse_args()

    # ── Print patch and exit ──────────────────────────────────────────────────
    if args.print_patch:
        print("=" * 72)
        print("  ADD THIS CODE TO vlc_gesture_control.py")
        print("=" * 72)
        print(PATCH_TEXT)
        print("=" * 72)
        return

    # ── OpenCV window setup ───────────────────────────────────────────────────
    win_name = "Gesture Guide  –  ARM SoC HCI Demo  [Q to close]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                               cv2.WINDOW_FULLSCREEN)

    print(f"[GUIDE] Watching state file: {STATE_FILE}")
    print("[GUIDE] Press Q in the guide window to close.")
    print(f"[GUIDE] Run with --print-patch to see what to add to vlc_gesture_control.py")
    print()

    last_action_time = 0.0
    prev_action_msg  = ""

    while True:
        state = read_state(STATE_FILE)

        # Track when a new action fires (for banner fade timer)
        current_action = state.get("action", "")
        if current_action and current_action != prev_action_msg:
            last_action_time = time.time()
            prev_action_msg  = current_action

        frame = build_frame(state, last_action_time)
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(int(POLL_INTERVAL * 1000)) & 0xFF
        if key == ord('q') or key == 27:   # Q or Escape
            break

    cv2.destroyAllWindows()
    print("[GUIDE] Closed.")


if __name__ == "__main__":
    main()
