"""Fast-test tracking bench (point 6). Replays BoT-SORT over a cached segment
under several configs and scores each against LTPI GT with standard MOT metrics
(IDF1, MOTA, ID-switches, fragmentations, mostly-tracked). GT identity is the
persistent player id (team*100+jersey), so IDF1/IDsw measure exactly the
identity-consistency the pipeline needs. Baselines: shipped vs OSNet embedder."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import cv2
import numpy as _np
if not hasattr(_np,'asfarray'): _np.asfarray=lambda a,dtype=_np.float64: _np.asarray(a,dtype=dtype)
import motmetrics as mm
import numpy as np
import pandas as pd

from app.services.detector import Detection
from app.services.tracker import BoxmotTracker

GT = "/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test/2/subtracks.csv"

CONFIGS = [
    ("shipped",      dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90)),
    ("strict-match", dict(appearance_thresh=0.40, match_thresh=0.65, proximity_thresh=0.50, track_buffer=90)),
    ("strict-all",   dict(appearance_thresh=0.25, match_thresh=0.65, proximity_thresh=0.60, track_buffer=30)),
    ("very-strict",  dict(appearance_thresh=0.18, match_thresh=0.55, proximity_thresh=0.60, track_buffer=20)),
]


def replay(cache, video, params):
    trk = BoxmotTracker(frame_rate=30, with_reid=True, **params)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cache[0][0])
    obs = []
    for fi, dets_np, embs in cache:
        ok, frame = cap.read()
        if not ok:
            break
        dets = [Detection(bbox=(r[0], r[1], r[2], r[3]), class_id=int(r[5]), class_name="", confidence=float(r[4])) for r in dets_np]
        use_emb = embs if len(embs) and embs.shape[1] > 1 else None
        for det, tid in trk.update(dets, frame, embeddings=use_emb):
            if tid is not None and det.class_id == 0:
                obs.append((fi, int(tid), *det.bbox))
    cap.release()
    return pd.DataFrame(obs, columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2"])


def _xywh(df):
    return np.c_[df.x1, df.y1, df.x2 - df.x1, df.y2 - df.y1]


def score_mot(obs, gt):
    acc = mm.MOTAccumulator(auto_id=False)
    obs = obs.copy()
    obs["image_id"] = obs.frame_idx + 1
    obs_by = {f: g for f, g in obs.groupby("image_id")}
    for img, gtf in gt.groupby("image_id"):
        hyp = obs_by.get(img)
        gboxes, hboxes = _xywh(gtf), (_xywh(hyp) if hyp is not None else np.zeros((0, 4)))
        dist = mm.distances.iou_matrix(gboxes, hboxes, max_iou=0.5)
        acc.update(gtf.id.tolist(), (hyp.track_id.tolist() if hyp is not None else []), dist, frameid=img)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=["idf1", "idp", "idr", "mota", "num_switches",
                                    "num_fragmentations", "mostly_tracked", "mostly_lost",
                                    "num_unique_objects"], name="x").iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT)
    gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]

    print(f"# {args.label}  segment {cache[0][0]}..{cache[-1][0]}  GT: {gt.id.nunique()} ids, {len(gt)} boxes")
    print(f"{'config':<13}{'IDF1':>7}{'MOTA':>7}{'IDsw':>6}{'Frag':>6}{'MT':>4}{'ML':>4}")
    for name, params in CONFIGS:
        s = score_mot(replay(cache, args.video, params), gt)
        print(f"{name:<13}{s.idf1:>7.3f}{s.mota:>7.3f}{int(s.num_switches):>6d}{int(s.num_fragmentations):>6d}"
              f"{int(s.mostly_tracked):>4d}{int(s.mostly_lost):>4d}", flush=True)


if __name__ == "__main__":
    main()
