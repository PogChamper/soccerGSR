"""Per-frame image->world homography grid for a segment, for the B5 pitch-coord
reachability gate. Mirrors ltpi-research extract_pitch.py but stores H_i2w per
frame (not per GT subtrack). Removes camera panning: two fragments of one player
at reachable PITCH-METRE positions can be merged even when appearance is unsure —
the fix for image-space motion failing on the panning camera."""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
os.environ.setdefault("MODEL_AUTO_DOWNLOAD", "false")

import cv2
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from app.utils.cuda_env import bootstrap
    bootstrap()
    from app.services.keypoints import HRNetKeypointsExtractor
    from app.services.calibration import PnLCalibrator
    kp = HRNetKeypointsExtractor()
    calib = PnLCalibrator(1920, 1080)

    cap = cv2.VideoCapture(args.video)
    Hs, n_ok = {}, 0
    frames = list(range(args.lo, args.hi + 1, args.stride))
    for fi in tqdm(frames, desc=f"H {args.lo}-{args.hi}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            res = calib.calibrate_one(*kp.extract(frame))
        except Exception:
            res = None
        if res is None:
            continue
        Hs[fi] = np.asarray(res["H_i2w"], dtype=np.float64)
        n_ok += 1
    cap.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(Hs, f)
    print(f"[H] {n_ok}/{len(frames)} calibrated -> {args.out}")


if __name__ == "__main__":
    main()
