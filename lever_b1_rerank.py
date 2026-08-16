"""B1 — k-reciprocal re-ranking of the tracklet-tracklet appearance distance in
the offline relink. Instead of raw cosine between tracklet OSNet centroids, use
the transductive k-reciprocal Jaccard distance (advanced.py, our identification
breakthrough lever) computed over ALL tracklets, then greedily merge under the
temporal mutual-exclusion constraint. Scored IDF1 vs the raw-cosine relink."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np
import pandas as pd

from lever1_break import occlusion_break
from lever5_relink import attach_emb
from track_bench import GT, replay, score_mot
from ltpi_research.advanced import k_reciprocal_rerank


def tracklets(obs, emb):
    tl = []
    for tid, g in obs.groupby("track_id"):
        idx = g.index.to_numpy()
        c = emb[idx].mean(0)
        c /= max(np.linalg.norm(c), 1e-9)
        tl.append((tid, c, int(g.frame_idx.min()), int(g.frame_idx.max()), len(g)))
    return tl


def greedy(order, sim, spans, thresh):
    """Greedy longest-first clustering under temporal mutual-exclusion, scoring a
    tracklet against a cluster by MAX pairwise similarity to its members
    (max-linkage) over the given tracklet-tracklet similarity matrix ``sim``."""
    clusters = []
    assign = {}
    for i in order:
        f0, f1 = spans[i]
        best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["spans"]):
                continue
            s = max(sim[i, m] for m in cl["members"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= thresh:
            clusters[best]["members"].append(i)
            clusters[best]["spans"].append((f0, f1))
            assign[i] = best
        else:
            clusters.append(dict(members=[i], spans=[(f0, f1)]))
            assign[i] = len(clusters) - 1
    return assign, len(clusters)


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
    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs).reset_index(drop=True)
    emb = attach_emb(obs, cache)[broken.orig.to_numpy()]
    broken = broken.reset_index(drop=True)

    tl = tracklets(broken, emb)
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]
    order = sorted(range(len(tl)), key=lambda i: -tl[i][4])
    tid_of = [t[0] for t in tl]

    cos = cents @ cents.T  # cosine sim (unit centroids)
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)  # transductive

    def run(sim, grid, tag):
        for th in grid:
            a, k = greedy(order, sim, spans, th)
            rel = broken.copy(); rel["track_id"] = broken.track_id.map({tid_of[i]: a[i] for i in a})
            s = score_mot(rel, gt)
            print(f"{'%s th=%.2f' % (tag, th):<22}{s.idf1:>7.3f}{int(s.num_switches):>6d}{k:>7d}", flush=True)

    print(f"{'variant':<22}{'IDF1':>7}{'IDsw':>6}{'#clust':>7}")
    run(cos, (0.60, 0.68, 0.74, 0.80, 0.86), "cosine")
    run(rr, (0.30, 0.40, 0.50, 0.60, 0.70), "k-recip")


if __name__ == "__main__":
    main()
