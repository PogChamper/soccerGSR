"""Replay BoT-SORT over cached detections+embeddings under different association
thresholds and score the resulting tracks against LTPI GT (segment): ID-switch
rate, track purity, fragmentation, recall. Isolates whether tuning association
cuts the 44 % ID-switch rate — the pipeline's end-to-end bottleneck."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import cv2
import numpy as np
import pandas as pd

from app.services.detector import Detection
from app.services.tracker import BoxmotTracker

GT = "/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test/2/subtracks.csv"

CONFIGS = [
    ("shipped",         dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90)),
    ("strict-appear",   dict(appearance_thresh=0.25, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90)),
    ("strict-match",    dict(appearance_thresh=0.40, match_thresh=0.65, proximity_thresh=0.50, track_buffer=90)),
    ("short-buffer",    dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=30)),
    ("strict-all",      dict(appearance_thresh=0.25, match_thresh=0.65, proximity_thresh=0.60, track_buffer=30)),
    ("very-strict",     dict(appearance_thresh=0.18, match_thresh=0.55, proximity_thresh=0.60, track_buffer=20)),
    ("looser",          dict(appearance_thresh=0.50, match_thresh=0.90, proximity_thresh=0.40, track_buffer=150)),
]


def iou_to(gb, box):
    ix1, iy1 = np.maximum(gb[:, 0], box[0]), np.maximum(gb[:, 1], box[1])
    ix2, iy2 = np.minimum(gb[:, 2], box[2]), np.minimum(gb[:, 3], box[3])
    iw, ih = np.clip(ix2 - ix1, 0, None), np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


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
        tracked = trk.update(dets, frame, embeddings=embs if len(embs) and embs.shape[1] > 1 else None)
        for det, tid in tracked:
            if tid is not None and det.class_id == 0:
                obs.append((fi, int(tid), *det.bbox))
    cap.release()
    return pd.DataFrame(obs, columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2"])


def evaluate(obs, gt):
    obs = obs.copy()
    obs["image_id"] = obs.frame_idx + 1
    gt_by_frame = {f: g for f, g in gt.groupby("image_id")}
    rows, matched, total = [], 0, 0
    for img, g in obs.groupby("image_id"):
        gtf = gt_by_frame.get(img)
        if gtf is None:
            continue
        total += len(gtf)
        gb = gtf[["x1", "y1", "x2", "y2"]].to_numpy(float)
        gid = gtf["identity"].to_numpy()
        used = np.zeros(len(gtf), bool)
        for _, o in g.iterrows():
            ious = iou_to(gb, np.array([o.x1, o.y1, o.x2, o.y2], float))
            j = int(ious.argmax())
            if ious[j] >= 0.5 and not used[j]:
                used[j] = True
                rows.append((int(o.track_id), int(gid[j])))
        matched += used.sum()
    m = pd.DataFrame(rows, columns=["app_track", "gt_id"])
    switches = sum(g.gt_id.nunique() > 1 for _, g in m.groupby("app_track"))
    purity = np.mean([g.gt_id.value_counts().iloc[0] / len(g) for _, g in m.groupby("app_track")])
    frag = m.groupby("gt_id").app_track.nunique().mean()
    ntr = m.app_track.nunique()
    return dict(recall=matched / max(total, 1), n_tracks=ntr, switch_pct=switches / max(ntr, 1),
                purity=purity, frag=frag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT)
    gt["identity"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]

    print(f"segment frames {cache[0][0]}..{cache[-1][0]}  ({len(cache)} frames)")
    print(f"{'config':<14}{'recall':>8}{'#tracks':>8}{'switch%':>9}{'purity':>8}{'frag':>7}")
    for name, params in CONFIGS:
        obs = replay(cache, args.video, params)
        r = evaluate(obs, gt)
        print(f"{name:<14}{r['recall']:>8.3f}{r['n_tracks']:>8d}{r['switch_pct']*100:>8.0f}%{r['purity']:>8.3f}{r['frag']:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
