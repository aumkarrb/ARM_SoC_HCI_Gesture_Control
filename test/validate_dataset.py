#!/usr/bin/env python3
"""
IPD Dataset Validator
=====================
Checks everything BEFORE you run process_ipd_trtpose.py
Catches: missing files, broken videos, annotation mismatches, class imbalance

Run:
    python3 validate_dataset.py --ipd_dir "$IPD_DIR"
"""

import os, sys, cv2, csv, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Target gestures we care about ────────────────────────────────────────────
TARGET_GESTURES = {
    'G01': 'click_one',
    'G02': 'click_two',
    'G05': 'throw_left',
    'G06': 'throw_right',
    'G10': 'zoom_in',
    'G11': 'zoom_out',
}

ALL_GESTURE_CODES = {
    'D0X','B0A','B0B','G01','G02','G03','G04',
    'G05','G06','G07','G08','G09','G10','G11'
}

PASS = "  ✅"
WARN = "  ⚠️ "
FAIL = "  ❌"

issues   = []
warnings = []

def ok(msg):   print(f"{PASS} {msg}")
def warn(msg): print(f"{WARN} {msg}"); warnings.append(msg)
def fail(msg): print(f"{FAIL} {msg}"); issues.append(msg)


# ─────────────────────────────────────────────────────────────────────────────

def validate(ipd_dir, sample_videos=3):
    root = Path(ipd_dir)

    print("\n" + "="*60)
    print("  IPD Dataset Validator")
    print("="*60)
    print(f"  Path: {root}\n")

    # ── 1. Root structure ────────────────────────────────────────────────
    print("[1] Root structure")
    if not root.exists():
        fail(f"Dataset root not found: {root}")
        print("\n❌ Cannot continue — path doesn't exist.")
        sys.exit(1)

    has_videos      = (root/'videos').exists()
    has_annotations = (root/'annotations').exists()
    has_metadata    = (root/'metadata.csv').exists()

    ok("Dataset root found") if True else None
    ok("videos/ folder")      if has_videos      else fail("videos/ folder missing")
    ok("annotations/ folder") if has_annotations else fail("annotations/ folder missing")
    ok("metadata.csv")        if has_metadata     else warn("metadata.csv missing (not critical)")

    # ── 2. Videos ────────────────────────────────────────────────────────
    print("\n[2] Video files")
    video_dir = None
    for candidate in [root/'videos'/'videos', root/'videos']:
        avi_list = list(candidate.glob('*.avi')) if candidate.exists() else []
        if avi_list:
            video_dir = candidate
            break

    if video_dir is None:
        fail("No .avi files found under videos/")
    else:
        avi_files = sorted(video_dir.glob('*.avi'))
        ok(f"{len(avi_files)} .avi files in {video_dir.relative_to(root)}")

        # Check a sample of videos can actually be opened
        print(f"  Sampling {sample_videos} videos for readability...")
        broken = []
        for vf in avi_files[:sample_videos]:
            cap = cv2.VideoCapture(str(vf))
            if not cap.isOpened():
                broken.append(vf.name)
            else:
                ret, _ = cap.read()
                if not ret:
                    broken.append(vf.name)
            cap.release()

        if broken:
            fail(f"{len(broken)} videos unreadable: {broken}")
        else:
            # Get stats on a good video
            cap = cv2.VideoCapture(str(avi_files[0]))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            ok(f"Videos readable  |  {width}×{height}  {fps:.0f}fps  ~{frames} frames each")

    # ── 3. Annotation files ──────────────────────────────────────────────
    print("\n[3] Annotation files")
    annot_dir = None
    for candidate in [root/'annotations'/'annotations', root/'annotations']:
        if (candidate/'Annot_TrainList.txt').exists():
            annot_dir = candidate
            break

    needed = ['Annot_TrainList.txt', 'Annot_TestList.txt',
              'class_details.txt', 'classIdx.txt']
    if annot_dir is None:
        fail("Annot_TrainList.txt not found in annotations/")
    else:
        ok(f"Annotation dir: {annot_dir.relative_to(root)}")
        for f in needed:
            path = annot_dir / f
            if path.exists():
                lines = len(path.read_text().splitlines())
                ok(f"{f}  ({lines} lines)")
            else:
                warn(f"{f} missing") if f != 'Annot_TrainList.txt' \
                    else fail(f"{f} MISSING — required!")

    # ── 4. Parse Annot_TrainList ─────────────────────────────────────────
    print("\n[4] Annotation content (Annot_TrainList.txt)")
    annot_file = annot_dir / 'Annot_TrainList.txt' if annot_dir else None

    if annot_file and annot_file.exists():
        per_gesture   = defaultdict(int)    # gesture_code → total segments
        per_video     = defaultdict(set)    # video → gesture codes seen
        bad_lines     = []
        total_lines   = 0
        unknown_codes = set()

        with open(annot_file) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split(',')
                if len(parts) < 5:
                    bad_lines.append(ln); continue

                video_name = parts[0].strip()
                code       = parts[1].strip()
                try:
                    start_f = int(parts[3])
                    end_f   = int(parts[4])
                except ValueError:
                    bad_lines.append(ln); continue

                total_lines += 1
                per_gesture[code] += 1
                per_video[video_name].add(code)

                if code not in ALL_GESTURE_CODES:
                    unknown_codes.add(code)

                # Check frame order
                if end_f < start_f:
                    bad_lines.append(ln)

        ok(f"{total_lines} valid annotation entries")
        ok(f"{len(per_video)} unique videos annotated")

        if bad_lines:
            warn(f"{len(bad_lines)} malformed lines (will be skipped): {bad_lines[:5]}")
        if unknown_codes:
            warn(f"Unknown gesture codes: {unknown_codes}")
        else:
            ok("All gesture codes recognised")

        # ── 5. Target gesture coverage ────────────────────────────────────
        print("\n[5] Target gesture coverage (G01 G02 G05 G06 G10 G11)")

        target_counts = {}
        for code in TARGET_GESTURES:
            n = per_gesture.get(code, 0)
            target_counts[code] = n

        all_found = True
        for code, name in TARGET_GESTURES.items():
            n = target_counts[code]
            label = f"{code} ({name}): {n} segments"
            if n == 0:
                fail(f"{label}")
                all_found = False
            elif n < 20:
                warn(f"{label}  ← very few, may affect accuracy")
            else:
                ok(label)

        # Imbalance check
        counts = list(target_counts.values())
        counts_nonzero = [c for c in counts if c > 0]
        if counts_nonzero:
            ratio = max(counts_nonzero) / (min(counts_nonzero) + 1e-9)
            if ratio > 5:
                warn(f"Severe class imbalance: {max(counts_nonzero)}x vs {min(counts_nonzero)}x "
                     f"(ratio {ratio:.1f}). Class weights will compensate.")
            elif ratio > 2:
                warn(f"Moderate class imbalance (ratio {ratio:.1f}). Class weights applied automatically.")
            else:
                ok(f"Class balance OK (ratio {ratio:.1f})")

        # ── 6. Video ↔ annotation cross-check ────────────────────────────
        print("\n[6] Video ↔ annotation cross-check")
        if video_dir:
            avi_stems  = {f.stem for f in video_dir.glob('*.avi')}
            annot_vids = set(per_video.keys())

            # Only check videos that have TARGET gestures
            target_annot_vids = set()
            with open(annot_file) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[1].strip() in TARGET_GESTURES:
                        target_annot_vids.add(parts[0].strip())

            missing_vids = target_annot_vids - avi_stems
            extra_vids   = avi_stems - annot_vids

            ok(f"{len(avi_stems)} .avi files,  {len(target_annot_vids)} annotated with target gestures")

            if missing_vids:
                warn(f"{len(missing_vids)} annotated videos have NO .avi file "
                     f"(will be skipped): {list(missing_vids)[:3]}...")
                coverage = (len(target_annot_vids) - len(missing_vids)) / len(target_annot_vids) * 100
                warn(f"Effective coverage: {coverage:.0f}% of annotated target-gesture videos have video files")
            else:
                ok(f"All annotated target-gesture videos have .avi files ✅")

        # ── 7. Sequence estimate ──────────────────────────────────────────
        print("\n[7] Expected sequence counts after processing")

        total_target_segments = sum(target_counts.values())
        # Rough estimate: ~70% detection rate with MediaPipe/trt_pose
        est_seqs = int(total_target_segments * 0.70)
        est_train = int(est_seqs * 0.70)
        est_val   = int(est_seqs * 0.10)
        est_test  = int(est_seqs * 0.20)

        print(f"  Total target segments in annotations : {total_target_segments}")
        print(f"  Estimated sequences after extraction : ~{est_seqs}  (70% detection rate)")
        print(f"    train (~70%) : ~{est_train}")
        print(f"    val   (~10%) : ~{est_val}")
        print(f"    test  (~20%) : ~{est_test}")
        print(f"    per class    : ~{est_train // len(TARGET_GESTURES)}")

        if est_train < 200:
            warn(f"⚠️  Very small training set (~{est_train} seqs). "
                 f"Will auto-use LightweightCNNLSTM in train.py")
        elif est_train < 400:
            warn(f"Small-ish training set (~{est_train} seqs). "
                 f"train.py will auto-switch to LightweightCNNLSTM")
        else:
            ok(f"Training set size looks good (~{est_train} seqs) → full GestureCNNLSTM")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  VALIDATION SUMMARY")
    print("="*60)
    print(f"  ✅ Passed  : {20 - len(issues) - len(warnings)} checks")
    print(f"  ⚠️  Warnings: {len(warnings)}")
    print(f"  ❌ Errors  : {len(issues)}")

    if issues:
        print(f"\n  Fix these before processing:")
        for i in issues: print(f"    • {i}")
    elif warnings:
        print(f"\n  Warnings (non-blocking):")
        for w in warnings: print(f"    • {w}")
        print(f"\n  ✅ Dataset OK to process despite warnings")
    else:
        print(f"\n  🟢 Dataset looks clean — ready to process!")

    print(f"\n  NEXT COMMAND:")
    print(f"  python3 process_ipd_trtpose.py \\")
    print(f'      --ipd_dir "{ipd_dir}" \\')
    print(f"      --output_dir ipd_processed \\")
    print(f"      --seq_len 30 \\")
    print(f"      --max_videos 5   # test run first")
    print()

    return len(issues) == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ipd_dir', required=True,
                   help='Path to IPD dataset root')
    p.add_argument('--sample_videos', type=int, default=3,
                   help='How many videos to test-open (default: 3)')
    args = p.parse_args()
    ok_result = validate(args.ipd_dir, args.sample_videos)
    sys.exit(0 if ok_result else 1)