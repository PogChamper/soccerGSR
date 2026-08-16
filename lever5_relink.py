"""Levers 1+5 — split-then-relink. Break tracks into occlusion-free tracklets
(lever 1), then globally re-link tracklets into identities by OSNet appearance
under a temporal mutual-exclusion constraint (lever 5): a greedy, length-ordered
constrained clustering. Scored IDF1/IDsw vs the shipped baseline on the subset."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import numpy as np
import pandas as pd

from lever1_break import occlusion_break
from track_bench import GT, replay, score_mot


def attach_emb(obs, cache):
    """Recover each obs box's OSNet embedding from the cache by (frame, x1, y1)."""
    look = {}
    for fi, dets_np, embs in cache:
        for j, r in enumerate(dets_np):
            look[(fi, round(float(r[0]), 1), round(float(r[1]), 1))] = embs[j]
    D = cache[0][2].shape[1]
    arr = np.zeros((len(obs), D), np.float32)
    for i, (fi, x1, y1) in enumerate(zip(obs.frame_idx, obs.x1, obs.y1)):
        e = look.get((int(fi), round(float(x1), 1), round(float(y1), 1)))
        if e is not None:
            arr[i] = e
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(n, 1e-9)


def relink(obs, emb, sim_thresh, team_gate=False, area_weight=False):
    """Greedy constrained clustering of tracklets -> identities.
    Lever 3: area-weighted centroids (big crops = cleaner). Lever 4: forbid
    merging tracklets of different predicted teams (2-means over centroids)."""
    tl = []
    for tid, g in obs.groupby("track_id"):
        idx = g.index.to_numpy()
        w = ((g.x2 - g.x1) * (g.y2 - g.y1)).to_numpy(float) if area_weight else np.ones(len(g))
        c = (emb[idx] * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        c /= max(np.linalg.norm(c), 1e-9)
        tl.append((tid, c, int(g.frame_idx.min()), int(g.frame_idx.max()), len(g)))
    team = {}
    if team_gate:
        from sklearn.cluster import KMeans
        cents = np.array([t[1] for t in tl])
        lab = KMeans(2, n_init=5, random_state=0).fit_predict(cents)
        team = {tl[i][0]: int(lab[i]) for i in range(len(tl))}
    tl.sort(key=lambda t: -t[4])  # longest first
    clusters = []  # {'cent','n','spans','team'}
    assign = {}
    for tid, c, f0, f1, n in tl:
        best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if team_gate and cl["team"] != team[tid]:
                continue
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["spans"]):
                continue
            s = float(c @ cl["cent"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= sim_thresh:
            cl = clusters[best]
            cl["cent"] = cl["cent"] * cl["n"] + c * n
            cl["cent"] /= max(np.linalg.norm(cl["cent"]), 1e-9)
            cl["n"] += n
            cl["spans"].append((f0, f1))
            assign[tid] = best
        else:
            clusters.append(dict(cent=c.copy(), n=n, spans=[(f0, f1)], team=team.get(tid, 0)))
            assign[tid] = len(clusters) - 1
    out = obs.copy()
    out["track_id"] = out.track_id.map(assign)
    return out, len(clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT)
    gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]

    obs = replay(cache, args.video, dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90))
    b = score_mot(obs, gt)
    print(f"{'variant':<20}{'IDF1':>7}{'MOTA':>7}{'IDsw':>6}{'#trk':>6}")
    print(f"{'baseline':<20}{b.idf1:>7.3f}{b.mota:>7.3f}{int(b.num_switches):>6d}{obs.track_id.nunique():>6d}", flush=True)

    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs)
    emb = attach_emb(obs, cache)
    be = emb[broken.orig.to_numpy()]
    for label, kw in [("relink", {}), ("+team+area (3,4)", dict(team_gate=True, area_weight=True))]:
        for th in (0.68, 0.74, 0.80, 0.86):
            rel, k = relink(broken, be, th, **kw)
            s = score_mot(rel[["frame_idx", "track_id", "x1", "y1", "x2", "y2"]], gt)
            print(f"{'%s th=%.2f' % (label, th):<20}{s.idf1:>7.3f}{s.mota:>7.3f}{int(s.num_switches):>6d}{k:>6d}", flush=True)


if __name__ == "__main__":
    main()
