"""Re-embed the cached detection boxes with the strong sports ReID (OSNet-
SoccerNet) instead of DINOv3, so the tracker sweep can test whether a
teammate-discriminative appearance model cuts BoT-SORT ID-switches. Reuses the
cached boxes (no re-detection); only the embeddings change."""
import argparse
import pickle
import sys
import time

sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

import cv2
import numpy as np

from extract_reid_torch import build_osnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--in-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="/home/dxdxxd/projects/dataIntegratorSoccer/models/osnet_x1_0_soccernet.pt")
    args = ap.parse_args()

    with open(args.in_cache, "rb") as f:
        cache = pickle.load(f)
    emb = build_osnet(args.model_path, device="cuda")
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cache[0][0])

    out = []
    t0 = time.time()
    for i, (fi, dets_np, _) in enumerate(cache):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        crops = []
        for r in dets_np:
            x1, y1, x2, y2 = int(max(0, r[0])), int(max(0, r[1])), int(min(w, r[2])), int(min(h, r[3]))
            crops.append(frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((2, 2, 3), np.uint8))
        embs = emb.embed(crops).astype(np.float32) if crops else np.zeros((0, emb.embed_dim), np.float32)
        out.append((fi, dets_np, embs))
        if i % 500 == 0:
            print(f"  {i}/{len(cache)}  {i/max(time.time()-t0,1e-9):.1f} fps", flush=True)
    cap.release()
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"re-embedded {len(out)} frames (OSNet) -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
