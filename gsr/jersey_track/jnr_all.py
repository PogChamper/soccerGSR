"""Label-free uncertainty-JNR extraction over ALL player/gk detections of a split.

One pass per sequence, no GT file opened: detections come from det.pkl, gate/read
columns from jersey.pkl, frame files from the frame index (img1/%06d.jpg). Output
format matches jnr_valid12.py (csv + npz with probs/uncertainty/convnext_logits),
so build_jnr_cache.py consumes it unchanged. This is the extraction path that may
run on test: jnr_valid12.py IoU-matches against the split's labels and must not.

Env: /home/dxdxxd/projects/soccer/sn-gamestate/.venv (torch 1.13, timm 1.0.15).
    python gsr/jersey_track/jnr_all.py --split valid --seqs SNGS-021 --out <dir>
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr/jersey_track")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

import cv2
import numpy as np
import pandas as pd

import common as C
from extract_uncertainty_jnr import DEFAULT_CHECKPOINT, build_model, infer_batch


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def all_rows(split, seq):
    det = C.load(C.OUT_ROOT / split / seq / "det.pkl")
    jer = C.load(C.OUT_ROOT / split / seq / "jersey.pkl")
    rows = []
    for fi, (image_id, boxes, (vis, tens, units)) in enumerate(
            zip(det["image_ids"], det["frames"], jer["frames"])):
        if len(boxes) == 0:
            continue
        didx = np.where(np.isin(boxes[:, 5], (0, 1)))[0]
        pt, pu = softmax(tens), softmax(units)
        for d in didx:
            b = boxes[d]
            scored = vis[d] > -1e8
            rows.append({
                "seq": seq, "image_id": image_id, "frame_idx": fi,
                "file": f"{fi + 1:06d}.jpg", "det_idx": int(d),
                "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                "h_det": float(b[3] - b[1]), "w_det": float(b[2] - b[0]),
                "scored": bool(scored),
                "vis_p": float(1 / (1 + np.exp(-vis[d]))) if scored else np.nan,
                "pred": int(pu[d].argmax()) if pt[d].argmax() == 0 else int(pt[d].argmax() * 10 + pu[d].argmax()),
                "conf": float(min(pt[d].max(), pu[d].max())) if scored else np.nan,
            })
    return pd.DataFrame(rows)


def run_seq(split, seq, model, device, batch, out):
    done = out / f"{seq}.npz"
    if done.exists():
        print(f"[jnr-all {seq}] exists, skip", flush=True)
        return
    df = all_rows(split, seq)
    df = df[df["scored"]].reset_index(drop=True)
    sd = C.seq_dir(split, seq)
    jer = C.load(C.OUT_ROOT / split / seq / "jersey.pkl")["frames"]
    cnx = np.stack([np.concatenate([jer[f][1][d], jer[f][2][d]])
                    for f, d in zip(df["frame_idx"], df["det_idx"])]).astype(np.float32)
    probs = np.zeros((len(df), 100), np.float16)
    unc = np.zeros(len(df), np.float32)
    t0 = time.time()
    for fname, grp in df.groupby("file", sort=False):
        frame = cv2.imread(str(sd / "img1" / fname))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        crops, idx = [], []
        for i, r in grp.iterrows():
            x1, y1, x2, y2 = int(max(0, r.x1)), int(max(0, r.y1)), int(min(w, r.x2)), int(min(h, r.y2))
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crops.append(frame[y1:y2, x1:x2])
            idx.append(i)
        for s in range(0, len(crops), batch):
            p, _, u = infer_batch(model, crops[s:s + batch], device)
            probs[idx[s:s + batch]] = p.astype(np.float16)
            unc[idx[s:s + batch]] = u
    df.to_csv(out / f"{seq}.csv", index=False)
    np.savez_compressed(done, probs=probs, uncertainty=unc, convnext_logits=cnx)
    print(f"[jnr-all {seq}] {len(df)} crops in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seqs", required=True, help="comma list of sequence names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(Path(args.checkpoint), args.device)
    for seq in args.seqs.split(","):
        run_seq(args.split, seq, model, args.device, args.batch, out)


if __name__ == "__main__":
    main()
