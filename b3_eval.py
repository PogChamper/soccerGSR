"""B3 — DBSCAN appearance splitter on top of the occlusion break. Within each
occlusion-free tracklet, cluster the per-frame OSNet embeddings; if they form more
than one appearance mode, split (an ID-switch that had no clean geometric occlusion).
Then k-reciprocal relink. Scored on tune + held-out vs B1 (no DBSCAN)."""
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from lever1_break import occlusion_break
from lever5_relink import attach_emb
from lever_b1_rerank import greedy, tracklets
from phase1_eval import full_metrics
from track_bench import GT, replay
from ltpi_research.advanced import k_reciprocal_rerank

VIDEO = "/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test/2/video.mp4"


def dbscan_split(broken, emb, eps, min_samples=3):
    """Split each tracklet whose per-frame embeddings form >1 DBSCAN cluster."""
    broken = broken.reset_index(drop=True)
    new = np.empty(len(broken), np.int64)
    nxt = 0
    for tid, g in broken.groupby("track_id"):
        idx = g.index.to_numpy()
        if len(idx) < 2 * min_samples:
            new[idx] = nxt; nxt += 1; continue
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(emb[idx])
        # noise (-1) folded into the nearest non-noise label by order; relabel densely
        uniq = {}
        for j, l in enumerate(lab):
            key = int(l)
            if key not in uniq:
                uniq[key] = nxt; nxt += 1
            new[idx[j]] = uniq[key]
    broken = broken.copy(); broken["track_id"] = new
    return broken


def prep(cache_path, eps):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT); gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]
    obs = replay(cache, VIDEO, dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90))
    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs).reset_index(drop=True)
    emb = attach_emb(obs, cache)[broken.orig.to_numpy()]
    if eps is not None:
        broken = dbscan_split(broken, emb, eps)
    tl = tracklets(broken, emb)
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]
    order = sorted(range(len(tl)), key=lambda i: -tl[i][4])
    tid_of = [t[0] for t in tl]
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    return gt, broken, spans, order, tid_of, rr


def relink_score(P, th):
    gt, broken, spans, order, tid_of, rr = P
    a, k = greedy(order, rr, spans, th)
    rel = broken.copy(); rel["track_id"] = broken.track_id.map({tid_of[i]: a[i] for i in a})
    m = full_metrics(rel, gt); m["k"] = k
    return m


def main():
    tune = sys.argv[1]; held = sys.argv[2]
    print(f"{'variant':<20}{'IDF1':>7}{'HOTA':>7}{'AssPr':>7}{'#k':>5}")
    # B1 baseline (no dbscan), and dbscan at a couple eps; tune th on tune, report held
    for label, eps in [("B1 (no dbscan)", None), ("B3 eps=0.25", 0.25), ("B3 eps=0.35", 0.35)]:
        Pt = prep(tune, eps)
        best = max((0.24, 0.28, 0.32, 0.36), key=lambda th: relink_score(Pt, th)["IDF1"])
        mt = relink_score(Pt, best)
        Ph = prep(held, eps)
        mh = relink_score(Ph, best)
        print(f"{label + ' TUNE':<20}{mt['IDF1']:>7.3f}{mt['HOTA']:>7.3f}{mt['AssPr']:>7.3f}{mt['k']:>5d}", flush=True)
        print(f"{label + ' HELD':<20}{mh['IDF1']:>7.3f}{mh['HOTA']:>7.3f}{mh['AssPr']:>7.3f}{mh['k']:>5d}", flush=True)


if __name__ == "__main__":
    main()
