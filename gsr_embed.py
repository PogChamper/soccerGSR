"""Cross-match validation on SoccerNet-GSR. Stage 1 (dfine-reid env): for a GSR
sequence, take the GT player boxes as detections (isolates association from
detection), crop + OSNet-embed each, and dump a cache in the same format the
tracker replay expects: list of (frame_idx, dets_np[x1y1x2y2,conf,cls], embs).
Also writes gt.csv (frame_idx, track_id, x1,y1,x2,y2) for scoring."""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")
import cv2
import numpy as np
import pandas as pd

from extract_reid_torch import build_osnet

OSNET = "/home/dxdxxd/projects/dataIntegratorSoccer/models/osnet_x1_0_soccernet.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=OSNET)
    args = ap.parse_args()
    seq = Path(args.seq)
    d = json.load(open(seq / "Labels-GameState.json"))
    id2file = {im["image_id"]: im["file_name"] for im in d["images"]}
    by_frame = {}
    for a in d["annotations"]:
        if (a.get("attributes") or {}).get("role") != "player" or a.get("track_id") is None:
            continue
        b = a["bbox_image"]
        by_frame.setdefault(a["image_id"], []).append(
            (int(a["track_id"]), float(b["x"]), float(b["y"]), float(b["x"] + b["w"]), float(b["y"] + b["h"])))

    emb = build_osnet(args.model_path, device="cuda")
    cache, gt_rows = [], []
    for fi, (img_id, file_name) in enumerate(sorted(id2file.items())):
        frame = cv2.imread(str(seq / "img1" / file_name))
        if frame is None:
            continue
        anns = by_frame.get(img_id, [])
        if not anns:
            cache.append((fi, np.zeros((0, 6), np.float32), np.zeros((0, 1), np.float32)))
            continue
        h, w = frame.shape[:2]
        crops, dets = [], []
        for tid, x1, y1, x2, y2 in anns:
            xi1, yi1, xi2, yi2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            if xi2 - xi1 < 4 or yi2 - yi1 < 8:
                continue
            crops.append(frame[yi1:yi2, xi1:xi2]); dets.append((x1, y1, x2, y2, 0.99, 0))
            gt_rows.append((fi, tid, x1, y1, x2, y2))
        e = emb.embed(crops).astype(np.float32) if crops else np.zeros((0, emb.embed_dim), np.float32)
        cache.append((fi, np.array(dets, np.float32) if dets else np.zeros((0, 6), np.float32), e))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f)
    pd.DataFrame(gt_rows, columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2"]).to_csv(
        str(out).replace(".pkl", "_gt.csv"), index=False)
    print(f"[{seq.name}] {len(cache)} frames, {sum(len(c[1]) for c in cache)} boxes -> {out}")


if __name__ == "__main__":
    main()
