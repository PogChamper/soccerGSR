"""Cross-match validation stage 2 (soccer_eda env). Replay BoT-SORT over the GSR
cache (GT boxes + OSNet embs), apply the offline stack (occlusion-break + k-recip
relink), and score tracking IDF1/HOTA vs GSR GT tracks — raw tracker vs +stack.
Tests whether the Phase-0/1 offline stack generalises off the single LTPI match."""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import cv2
import motmetrics as mm
import pandas as pd

from app.services.detector import Detection
from app.services.tracker import BoxmotTracker
from hota import hota
from ltpi_research.advanced import k_reciprocal_rerank


def replay(cache, seq):
    files = sorted(json.load(open(seq / "Labels-GameState.json"))["images"], key=lambda im: im["image_id"])
    files = [im["file_name"] for im in files]
    trk = BoxmotTracker(frame_rate=25, with_reid=True)
    obs = []
    for fi, dets_np, embs in cache:
        frame = cv2.imread(str(seq / "img1" / files[fi])) if fi < len(files) else None
        if frame is None:
            frame = np.zeros((1080, 1920, 3), np.uint8)
        dets = [Detection(bbox=(r[0], r[1], r[2], r[3]), class_id=0, class_name="", confidence=float(r[4])) for r in dets_np]
        use = embs if len(embs) and embs.shape[1] > 1 else None
        for det, tid in trk.update(dets, frame, embeddings=use):
            if tid is not None:
                obs.append((fi, int(tid), *det.bbox))
    return pd.DataFrame(obs, columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2"])


def occlusion_break(obs, iou_thresh=0.35):
    obs = obs.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    occ = np.zeros(len(obs), bool)
    for _, g in obs.groupby("frame_idx"):
        idx = g.index.to_numpy(); b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        if len(b) < 2:
            continue
        x1 = np.maximum.outer(b[:, 0], b[:, 0]); y1 = np.maximum.outer(b[:, 1], b[:, 1])
        x2 = np.minimum.outer(b[:, 2], b[:, 2]); y2 = np.minimum.outer(b[:, 3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        np.fill_diagonal(iou, 0.0); occ[idx] = iou.max(axis=1) >= iou_thresh
    obs = obs[~occ].reset_index(drop=True)
    new = np.empty(len(obs), np.int64); nxt = 0
    for tid, g in obs.groupby("track_id"):
        fr = g.frame_idx.to_numpy(); cut = np.concatenate([[True], np.diff(fr) > 1])
        ids = nxt + np.cumsum(cut) - 1; new[g.index.to_numpy()] = ids; nxt = ids.max() + 1
    obs = obs.copy(); obs["track_id"] = new; return obs


def attach_emb(obs, cache):
    look = {}
    for fi, dets_np, embs in cache:
        for j, r in enumerate(dets_np):
            look[(fi, round(float(r[0]), 1), round(float(r[1]), 1))] = embs[j]
    D = next((c[2].shape[1] for c in cache if len(c[2])), 512)
    arr = np.zeros((len(obs), D), np.float32)
    for i, (fi, x1, y1) in enumerate(zip(obs.frame_idx, obs.x1, obs.y1)):
        e = look.get((int(fi), round(float(x1), 1), round(float(y1), 1)))
        if e is not None and len(e) == D:
            arr[i] = e
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(n, 1e-9)


def relink(obs, cache, th=0.20):
    obs = obs.reset_index(drop=True)
    emb = attach_emb(obs, cache)
    tl = []
    for tid, g in obs.groupby("track_id"):
        idx = g.index.to_numpy(); c = emb[idx].mean(0); c /= max(np.linalg.norm(c), 1e-9)
        tl.append((tid, c, int(g.frame_idx.min()), int(g.frame_idx.max())))
    if len(tl) < 2:
        return obs
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]; tid_of = [t[0] for t in tl]
    order = sorted(range(len(tl)), key=lambda i: spans[i][0] - spans[i][1])
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    clusters, assign = [], {}
    for i in order:
        f0, f1 = spans[i]; best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["m"]):
                continue
            s = max(rr[i, m] for m in cl["idx"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= th:
            clusters[best]["idx"].append(i); clusters[best]["m"].append((f0, f1)); assign[tid_of[i]] = best
        else:
            clusters.append(dict(idx=[i], m=[(f0, f1)])); assign[tid_of[i]] = len(clusters) - 1
    out = obs.copy(); out["track_id"] = obs.track_id.map(assign); return out


def score(obs, gt):
    acc = mm.MOTAccumulator(auto_id=False)
    ob = {f: g for f, g in obs.groupby("frame_idx")}
    for fi, gtf in gt.groupby("frame_idx"):
        h = ob.get(fi)
        gb = np.c_[gtf.x1, gtf.y1, gtf.x2 - gtf.x1, gtf.y2 - gtf.y1]
        hb = np.c_[h.x1, h.y1, h.x2 - h.x1, h.y2 - h.y1] if h is not None else np.zeros((0, 4))
        d = mm.distances.iou_matrix(gb, hb, max_iou=0.5)
        acc.update(gtf.track_id.tolist(), (h.track_id.tolist() if h is not None else []), d, frameid=fi)
    s = mm.metrics.create().compute(acc, metrics=["idf1", "num_switches"], name="x").iloc[0]
    ht = hota(gt.rename(columns={"track_id": "id", "frame_idx": "frame"}),
              obs.rename(columns={"frame_idx": "frame"}))
    return s.idf1, ht["HOTA"], ht["AssPr"], ht["AssRe"], int(s.num_switches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--seq", required=True)
    args = ap.parse_args()
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(args.gt)
    obs = replay(cache, Path(args.seq))
    r = score(obs, gt)
    s = score(relink(occlusion_break(obs), cache), gt)
    name = Path(args.seq).name
    print(f"{name} raw    IDF1={r[0]:.3f} HOTA={r[1]:.3f} AssPr={r[2]:.3f} AssRe={r[3]:.3f} IDsw={r[4]}", flush=True)
    print(f"{name} +stack IDF1={s[0]:.3f} HOTA={s[1]:.3f} AssPr={s[2]:.3f} AssRe={s[3]:.3f} IDsw={s[4]}", flush=True)


if __name__ == "__main__":
    main()
