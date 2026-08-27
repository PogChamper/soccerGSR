"""OSNet embeddings only (no PnLCalib) for the dev-40 fragment rail.

Same crops, model and output format as s2_embed_calib.py's emb.pkl, with the
calibration leg skipped: the fragment rail GT-matches in image space, so no
homography is needed. Resumable per clip (skip if emb.pkl exists).

    python s2_embed_only.py valid SNGS-039,SNGS-040
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

import cv2
import numpy as np

import common as C
from extract_reid_torch import build_osnet

OSNET = "/home/dxdxxd/projects/dataIntegratorSoccer/models/osnet_x1_0_soccernet.pt"


def process(split: str, seq: str, emb) -> None:
    out = C.OUT_ROOT / split / seq / "emb.pkl"
    if out.exists():
        print(f"[s2e {seq}] exists, skip", flush=True)
        return
    det = C.load(C.OUT_ROOT / split / seq / "det.pkl")
    sd = C.seq_dir(split, seq)
    frames = []
    for fi, boxes in enumerate(det["frames"]):
        if len(boxes) == 0:
            frames.append(np.zeros((0, 512), np.float32))
            continue
        frame = cv2.imread(str(sd / "img1" / f"{fi + 1:06d}.jpg"))
        h, w = frame.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
            crops.append(frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((2, 2, 3), np.uint8))
        frames.append(emb.embed(crops).astype(np.float32))
    C.dump({"frames": frames}, out)
    print(f"[s2e {seq}] embedded {sum(len(f) for f in frames)} boxes -> emb.pkl", flush=True)


def main() -> None:
    split, seqs = sys.argv[1], sys.argv[2].split(",")
    emb = build_osnet(OSNET, device="cuda")
    for seq in seqs:
        process(split, seq, emb)


if __name__ == "__main__":
    main()
