"""Cache the GPU-heavy, tracker-independent part (detections + DINOv3 embeddings)
for a frame window, so BoT-SORT association thresholds can be swept by replaying
only the tracker (cheap CPU) instead of re-running detection/embedding each time."""
import argparse
import pickle
import sys
import time

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import cv2
import numpy as np

from app.services.detector import get_detector
from app.services.embedder import get_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    det = get_detector()
    emb = get_embedder()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    per_frame = []
    t0 = time.time()
    for fi in range(args.start, args.end):
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        if dets:
            bboxes = [tuple(d.bbox) for d in dets]
            embs = np.asarray(emb.embed_boxes(frame, bboxes), dtype=np.float32)
            dets_np = np.array([[*d.bbox, d.confidence, d.class_id] for d in dets], dtype=np.float32)
        else:
            dets_np = np.zeros((0, 6), np.float32)
            embs = np.zeros((0, 1), np.float32)
        per_frame.append((fi, dets_np, embs))
        if (fi - args.start) % 500 == 0:
            print(f"  {fi-args.start}/{args.end-args.start}  {(fi-args.start)/max(time.time()-t0,1e-9):.1f} fps", flush=True)
    cap.release()
    with open(args.out, "wb") as f:
        pickle.dump(per_frame, f)
    print(f"cached {len(per_frame)} frames -> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
