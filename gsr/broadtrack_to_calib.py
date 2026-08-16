"""Convert BroadTrack per-frame camera records into our calib.pkl homographies.

BroadTrack emits {frame: camera_record} in the SoccerNet calibration parameter
format. Each record -> sn_calibration Camera -> to_homography() gives the pitch
(world, metres, centre origin) -> image homography; we invert it to get the
image -> pitch homography our pipeline projects with. Writes calib_<tag>.pkl in
the same {"frames": [H or None]} shape as s2's calib.pkl, aligned to frame index
(BroadTrack processes img1/%06d.jpg in number order == our GT image order).

    python gsr/broadtrack_to_calib.py --results <broadtrack results root> --split test [--tag bt]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer/sn-gamestate/plugins/calibration")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
import numpy as np

import common as C
from sn_calibration_baseline.camera import Camera

def record_to_H(rec):
    """BroadTrack's 'cp' is the SoccerNet camera-prediction format (FOV, pan/tilt/
    roll, position in metres). Build the sn_calibration Camera from it and return
    the image -> pitch homography = inv(pitch -> image)."""
    import math
    cp = rec["cp"]
    w = float(cp["sensorResolutionWidthPixels"])
    h = float(cp["sensorResolutionHeightPixels"])
    fx = (w / 2.0) / math.tan(math.radians(cp["horizontalFieldOfViewDegrees"]) / 2.0)
    params = {
        "principal_point": [w / 2.0, h / 2.0],
        "x_focal_length": fx, "y_focal_length": fx,
        "pan_degrees": cp["panDegrees"], "tilt_degrees": cp["tiltDegrees"],
        "roll_degrees": cp["rollDegrees"],
        "position_meters": [cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]],
        "radial_distortion": [0.0] * 6, "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0] * 4,
    }
    cam = Camera(int(w), int(h))
    cam.from_json_parameters(params)
    return np.linalg.inv(cam.to_homography())  # image -> pitch


def frame_key(k):
    digits = "".join(c for c in str(k) if c.isdigit())
    return int(digits) if digits else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="BroadTrack output root (<root>/SNGS-*/broadtrack.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="bt")
    ap.add_argument("--seqs", default="all")
    args = ap.parse_args()
    seqs = C.list_seqs(args.split) if args.seqs == "all" else args.seqs.split(",")

    for seq in seqs:
        bt_path = Path(args.results) / seq / "broadtrack.json"
        if not bt_path.exists():
            print(f"[{seq}] no broadtrack.json -> skip", flush=True)
            continue
        bt = json.load(open(bt_path))
        n_frames = len(C.frame_index(C.load_labels(args.split, seq))[0])
        keys = sorted(bt, key=frame_key)
        frames, nfail = [], 0
        for k in keys:
            try:
                frames.append(record_to_H(bt[k]))
            except Exception:
                frames.append(None); nfail += 1
        # pad/truncate to our frame count
        if len(frames) < n_frames:
            frames += [None] * (n_frames - len(frames))
        frames = frames[:n_frames]
        C.dump({"frames": frames}, C.OUT_ROOT / args.split / seq / f"calib_{args.tag}.pkl")
        print(f"[{seq}] {len(bt)} records -> calib_{args.tag}.pkl ({nfail} bad)", flush=True)


if __name__ == "__main__":
    main()
