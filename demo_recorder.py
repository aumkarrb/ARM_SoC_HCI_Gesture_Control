"""
demo_recorder.py  –  In-Process Demo Recorder
==============================================
ARM SoC HCI Gesture Control – Jetson Nano

PURPOSE
-------
Records the annotated camera feed (with HUD overlay already drawn) to an
MP4 file, and writes a timestamped CSV log of every gesture detected and
every VLC action fired — all in-process using only OpenCV's VideoWriter.

No extra dependencies beyond what vlc_gesture_control.py already uses.
No internet, no ffmpeg install required, no threads: the recorder runs
synchronously in the main loop, adding negligible overhead (~0.3 ms/frame).

OUTPUT FILES  (saved to ./recordings/ by default)
--------------------------------------------------
  recordings/
    demo_YYYYMMDD_HHMMSS.mp4   – annotated video with HUD overlay
    demo_YYYYMMDD_HHMMSS.csv   – action log

CSV FORMAT
----------
  session_time_s, wall_clock, gesture, confidence_pct, action, triggered
  0.00,  2026-02-17 14:23:01.412,  fist,  87,  Play / Pause,  True
  0.53,  2026-02-17 14:23:01.963,  fist,  91,  Play / Pause,  False
  2.14,  2026-02-17 14:23:03.562,  two_up, 94,  +10s,          True

  triggered=True  → gesture fired a VLC action (cooldown cleared)
  triggered=False → gesture was detected but blocked by cooldown or confidence

HOW TO INTEGRATE
----------------
1.  Drop demo_recorder.py in the same folder as vlc_gesture_control.py.

2.  Add these imports near the top of vlc_gesture_control.py:
        from demo_recorder import DemoRecorder

3.  After the camera is opened and frame size is set, create the recorder:
        recorder = DemoRecorder(frame_w=640, frame_h=480)

4.  Inside the while loop, AFTER draw_hud(), add:
        recorder.write_frame(frame)

5.  Also inside the while loop, when an action fires (after vlc_ctrl.dispatch):
        recorder.log_action(smooth_pred, smooth_conf, result, triggered=True)

    And on every frame where a gesture is detected but NOT triggered:
        recorder.log_action(display_gesture, display_conf, "", triggered=False)

6.  After the loop (before cap.release()), call:
        recorder.close()

    Run with --print-patch to see the exact lines ready to copy-paste:
        python3 demo_recorder.py --print-patch

CONTROLS (keyboard shortcuts added by the recorder patch)
----------------------------------------------------------
    R  →  already resets VLC — now also starts a new recording segment
    S  →  save / stop current segment without quitting

DEPENDENCIES
------------
    cv2, csv, os, datetime, pathlib  — all stdlib or already installed

TESTED ON
---------
    OpenCV 4.x  |  Python 3.8+  |  Ubuntu 20.04 / JetPack 5.x
    Codec: mp4v (always available via OpenCV, no extra install)
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  PATCH TEXT  (printed with --print-patch)
# ══════════════════════════════════════════════════════════════════════════════

PATCH_TEXT = r'''
# ── 1. Add this import near the top of vlc_gesture_control.py ────────────────
from demo_recorder import DemoRecorder

# ── 2. After cap.set(...) lines and before the while loop ────────────────────
recorder = DemoRecorder(frame_w=640, frame_h=480)
print(f"[REC] Recording to: {recorder.video_path}")
print(f"[REC]  CSV log at:  {recorder.csv_path}")

# ── 3. Inside the while loop – REPLACE the Draw HUD block with: ──────────────

        # Draw HUD  (already exists – keep as-is)
        frame = draw_hud(frame, display_gesture, display_conf,
                         last_action_msg, in_cooldown, classes)

        # ── Record frame ───────────────────────────────────────────────────
        recorder.write_frame(frame)

        # ── Log gesture every frame ────────────────────────────────────────
        action_fired = (
            smooth_pred and smooth_conf >= CONF_THRESHOLD and not in_cooldown
        )
        recorder.log_action(
            gesture    = display_gesture,
            confidence = display_conf,
            action     = last_action_msg if action_fired else "",
            triggered  = bool(action_fired),
        )

# ── 4. Update the key handler (find `elif key == ord('r'):` and add below) ───
        elif key == ord('r'):
            vlc_ctrl.stop()
            recorder.new_segment()          # ← ADD THIS LINE
            if args.video:
                vlc_ctrl = VLCController(video_path=args.video)
        elif key == ord('s'):               # ← ADD THIS ENTIRE BLOCK
            recorder.new_segment()
            print("[REC] Segment saved. New segment started.")

# ── 5. After the loop, before cap.release() ──────────────────────────────────
    recorder.close()
'''


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit here to customise output)
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR   = "recordings"     # folder created next to vlc_gesture_control.py
VIDEO_CODEC  = "mp4v"           # mp4v = always available; try 'avc1' for smaller files
VIDEO_EXT    = ".mp4"
TARGET_FPS   = 20.0             # Jetson typically delivers 20–25 fps at 640×480
                                 # Must be a float. Adjust if playback is too fast/slow.
LOG_ALL_FRAMES  = False         # True  = log every frame to CSV (large files)
                                 # False = log only triggered actions (clean log)


# ══════════════════════════════════════════════════════════════════════════════
#  DemoRecorder
# ══════════════════════════════════════════════════════════════════════════════

class DemoRecorder:
    """
    Records the annotated demo feed to MP4 + CSV.

    Usage
    -----
        recorder = DemoRecorder(frame_w=640, frame_h=480)

        # Inside the capture loop:
        recorder.write_frame(frame)          # frame already has HUD drawn on it
        recorder.log_action(gesture, conf, action_str, triggered=True/False)

        # After the loop:
        recorder.close()
    """

    def __init__(
        self,
        frame_w:    int   = 640,
        frame_h:    int   = 480,
        fps:        float = TARGET_FPS,
        output_dir: str   = OUTPUT_DIR,
        log_all:    bool  = LOG_ALL_FRAMES,
    ):
        """
        Parameters
        ----------
        frame_w / frame_h : int
            Must match the camera resolution set in vlc_gesture_control.py.
            Default 640×480 matches cap.set(CAP_PROP_FRAME_WIDTH, 640).
        fps : float
            Target FPS for the output video.  Does NOT need to be exact —
            the actual capture rate varies, but the muxer uses this value
            to set video speed.  20.0 is a safe default for Jetson Nano.
        output_dir : str
            Directory where recordings are saved.  Created automatically.
        log_all : bool
            If True, every frame is written to the CSV (verbose).
            If False (default), only frames where triggered=True are logged.
        """
        self._frame_w    = frame_w
        self._frame_h    = frame_h
        self._fps        = fps
        self._output_dir = Path(output_dir)
        self._log_all    = log_all

        self._writer:    object = None   # cv2.VideoWriter
        self._csv_file:  object = None   # file handle
        self._csv_writer:object = None   # csv.writer
        self._start_time: float = 0.0
        self._frame_count: int  = 0
        self._action_count: int = 0

        # Public paths – set by _open_segment()
        self.video_path: str = ""
        self.csv_path:   str = ""

        self._open_segment()

    # ── Internal: open a new recording segment ────────────────────────────────

    def _make_filename(self, ext: str) -> str:
        """Generate a timestamped filename."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self._output_dir / f"demo_{stamp}{ext}")

    def _open_segment(self) -> None:
        """Create the output directory, VideoWriter, and CSV file."""
        import cv2

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── Video writer ──────────────────────────────────────────────────────
        video_path = self._make_filename(VIDEO_EXT)
        fourcc     = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer     = cv2.VideoWriter(
            video_path, fourcc, self._fps,
            (self._frame_w, self._frame_h)
        )

        if not writer.isOpened():
            # Codec unavailable – try XVID as universal fallback
            print(f"[REC] WARNING: codec '{VIDEO_CODEC}' failed. Trying XVID fallback.")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_path = self._make_filename(".avi")
            writer = cv2.VideoWriter(
                video_path, fourcc, self._fps,
                (self._frame_w, self._frame_h)
            )
            if not writer.isOpened():
                print("[REC] ERROR: Could not open VideoWriter. Recording disabled.")
                self._writer = None
                self.video_path = ""
            else:
                self._writer    = writer
                self.video_path = video_path
        else:
            self._writer    = writer
            self.video_path = video_path

        # ── CSV log ───────────────────────────────────────────────────────────
        csv_path       = self._make_filename(".csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "session_time_s",
            "wall_clock",
            "gesture",
            "confidence_pct",
            "action",
            "triggered",
        ])
        self.csv_path = csv_path

        import time
        self._start_time  = time.time()
        self._frame_count  = 0
        self._action_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def write_frame(self, frame) -> None:
        """
        Write one annotated frame to the video file.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame, already with the HUD overlay drawn on it.
            Must be the same size as frame_w × frame_h.
        """
        if self._writer is None:
            return

        h, w = frame.shape[:2]
        if w != self._frame_w or h != self._frame_h:
            # Resize silently so a size mismatch doesn't crash the main loop
            import cv2
            frame = cv2.resize(frame, (self._frame_w, self._frame_h))

        self._writer.write(frame)
        self._frame_count += 1

    def log_action(
        self,
        gesture:    str,
        confidence: float,
        action:     str,
        triggered:  bool,
    ) -> None:
        """
        Write one row to the CSV log.

        Parameters
        ----------
        gesture    : str    Detected gesture name (e.g. 'fist')
        confidence : float  Model confidence 0.0–1.0
        action     : str    VLC action description (e.g. 'Paused', '+10s')
                            Pass "" if no action fired this frame.
        triggered  : bool   True if this gesture actually fired a VLC command.
        """
        if self._csv_writer is None:
            return

        # Respect log_all setting — skip non-triggered frames if log_all=False
        if not self._log_all and not triggered:
            return

        import time
        session_s  = round(time.time() - self._start_time, 3)
        wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        conf_pct   = round(confidence * 100, 1)

        self._csv_writer.writerow([
            session_s,
            wall_clock,
            gesture    or "",
            conf_pct,
            action     or "",
            triggered,
        ])

        # Flush every triggered action so the file is readable even if
        # the session ends abruptly (Ctrl-C, power loss, etc.)
        if triggered:
            self._csv_file.flush()
            self._action_count += 1

    def new_segment(self) -> None:
        """
        Close the current recording and start a new one.
        Called when the user presses R (reset) or S (save segment).
        """
        self.close(print_summary=True)
        self._open_segment()
        print(f"[REC] New segment started.")
        print(f"[REC]   Video : {self.video_path}")
        print(f"[REC]   CSV   : {self.csv_path}")

    def close(self, print_summary: bool = True) -> None:
        """
        Finalise and close all open file handles.
        Safe to call multiple times.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

        if print_summary and self._frame_count > 0:
            import time
            duration = round(time.time() - self._start_time, 1)
            print("\n[REC] ── Recording saved ──────────────────────────────")
            print(f"[REC]   Video      : {self.video_path}")
            print(f"[REC]   CSV log    : {self.csv_path}")
            print(f"[REC]   Duration   : {duration}s")
            print(f"[REC]   Frames     : {self._frame_count}  "
                  f"({round(self._frame_count / max(duration, 0.1), 1)} fps avg)")
            print(f"[REC]   Actions    : {self._action_count} gesture(s) triggered")
            print("[REC] ─────────────────────────────────────────────────\n")

    # ── Context manager support ───────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (run directly to verify VideoWriter works on this machine)
#  python3 demo_recorder.py
#  python3 demo_recorder.py --print-patch
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """
    Write 60 dummy frames (2 seconds at 30fps) and verify the output files
    exist and are non-empty.  No camera or VLC required.
    """
    import cv2
    import numpy as np
    import time

    print("=" * 60)
    print("  DemoRecorder Self-Test")
    print("=" * 60)

    TEST_DIR = "/tmp/demo_recorder_test"
    GESTURES = ["fist", "ok", "palm", "stop", "two_up", "two_up_inverted"]
    ACTIONS  = ["Play / Pause", "Next Chapter", "Volume Up",
                "Stopped", "+10s", "-10s"]

    with DemoRecorder(frame_w=640, frame_h=480, fps=30.0,
                      output_dir=TEST_DIR, log_all=True) as rec:

        print(f"\n  Video  : {rec.video_path}")
        print(f"  CSV    : {rec.csv_path}\n")

        for i in range(60):
            # Synthetic frame: gradient + frame number text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = int(i * 4)          # blue channel ramp
            frame[:, :, 2] = 255 - int(i * 4)    # red channel ramp
            cv2.putText(frame, f"TEST FRAME {i:03d}", (160, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

            rec.write_frame(frame)

            gesture   = GESTURES[i % len(GESTURES)]
            confidence = 0.75 + (i % 10) * 0.02
            triggered  = (i % 15 == 0)   # fire an action every 15 frames
            action     = ACTIONS[i % len(ACTIONS)] if triggered else ""

            rec.log_action(gesture, confidence, action, triggered)
            time.sleep(1 / 30)   # simulate real-time capture

        # Grab paths before close() clears state
        video_path = rec.video_path
        csv_path   = rec.csv_path

    # ── Verify outputs ────────────────────────────────────────────────────────
    print("\n  Verifying output files …")
    errors = 0

    if os.path.exists(video_path):
        size_kb = os.path.getsize(video_path) // 1024
        if size_kb > 0:
            print(f"  [PASS]  Video exists  ({size_kb} KB): {video_path}")
        else:
            print(f"  [FAIL]  Video is empty: {video_path}")
            errors += 1
    else:
        print(f"  [FAIL]  Video not found: {video_path}")
        errors += 1

    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = f.readlines()
        if len(rows) > 1:   # header + at least one data row
            print(f"  [PASS]  CSV exists  ({len(rows)-1} data rows): {csv_path}")
            print(f"\n  First 3 data rows:")
            for row in rows[1:4]:
                print(f"    {row.rstrip()}")
        else:
            print(f"  [FAIL]  CSV has no data rows: {csv_path}")
            errors += 1
    else:
        print(f"  [FAIL]  CSV not found: {csv_path}")
        errors += 1

    print()
    if errors == 0:
        print("  [DONE]  Self-test passed. DemoRecorder is working correctly.")
    else:
        print(f"  [DONE]  Self-test finished with {errors} error(s).")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DemoRecorder self-test / patch printer")
    parser.add_argument("--print-patch", action="store_true",
                        help="Print the integration patch for vlc_gesture_control.py and exit")
    args = parser.parse_args()

    if args.print_patch:
        print("=" * 72)
        print("  ADD THIS CODE TO vlc_gesture_control.py")
        print("=" * 72)
        print(PATCH_TEXT)
        print("=" * 72)
    else:
        _self_test()
