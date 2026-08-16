"""Stage 3a (soccer_eda env): BoT-SORT association -> track.pkl.

Split out from assembly because it is the only step that must read frames (for
ECC camera-motion compensation) and it is invariant to the identity/jersey/team
levers — so we run it once per tracker config and iterate assembly cheaply.

    python gsr/s3a_track.py <split> <seqs|all>
Tracker knobs via env: GSR_TRACK_BUFFER, GSR_NEW_THRESH, GSR_APP_THRESH.
"""
import os
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")

import cv2
import numpy as np

import common as C
from app.services.detector import Detection
from app.services.tracker import BoxmotTracker

BUFFER = int(os.environ.get("GSR_TRACK_BUFFER", "90"))
NEW_THRESH = float(os.environ.get("GSR_NEW_THRESH", "0.6"))
APP_THRESH = float(os.environ.get("GSR_APP_THRESH", "0.4"))
TRACK_TAG = os.environ.get("GSR_TRACK_TAG", "base")


def process(split, seq):
    base = C.OUT_ROOT / split / seq
    det = C.load(base / "det.pkl")
    emb = C.load(base / "emb.pkl")["frames"]
    _, files = C.frame_index(C.load_labels(split, seq))
    sd = C.seq_dir(split, seq)

    trk = BoxmotTracker(frame_rate=25, with_reid=True, track_buffer=BUFFER,
                        new_track_thresh=NEW_THRESH, appearance_thresh=APP_THRESH)
    records = []
    for fi, boxes in enumerate(det["frames"]):
        frame = cv2.imread(str(sd / "img1" / files[fi]))
        if frame is None:
            frame = np.zeros((1080, 1920, 3), np.uint8)
        dets = [Detection(bbox=(b[0], b[1], b[2], b[3]), class_id=int(b[5]),
                          class_name="", confidence=float(b[4])) for b in boxes]
        e = emb[fi] if len(emb[fi]) == len(dets) else None
        for j, (d, tid) in enumerate(trk.update(dets, frame, embeddings=e)):
            if tid is not None:
                records.append((fi, j, int(tid), int(boxes[j][5]),
                                tuple(float(x) for x in boxes[j][:4]), float(boxes[j][4])))
    C.dump({"records": records}, base / f"track_{TRACK_TAG}.pkl")
    ntr = len({r[2] for r in records})
    print(f"[s3a {seq}] {len(records)} tracked boxes, {ntr} tracks -> track_{TRACK_TAG}.pkl", flush=True)


def main():
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    for seq in seqs:
        process(split, seq)


if __name__ == "__main__":
    main()
