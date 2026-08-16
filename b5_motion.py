"""B5 (image-space proxy) — motion-continuity bridge in the relink. AssPr is
already 0.98; the headroom is AssRe (same player under-linked across gaps). For a
candidate merge, add a bonus when the two tracklets' nearer endpoints are close in
the image and the time gap is small (a reachable, continuous move) — so an
appearance-ambiguous same-player pair still links. A cheap proxy for world-coord
reachability (no calibration); if it helps, the pitch-coord version follows."""
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np
import pandas as pd

from lever1_break import occlusion_break
from lever5_relink import attach_emb
from phase1_eval import full_metrics
from track_bench import GT, replay
from ltpi_research.advanced import k_reciprocal_rerank

VIDEO = "/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test/2/video.mp4"


def prep(cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT); gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]
    obs = replay(cache, VIDEO, dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90))
    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs).reset_index(drop=True)
    emb = attach_emb(obs, cache)[broken.orig.to_numpy()]
    broken["fx"] = (broken.x1 + broken.x2) / 2; broken["fy"] = broken.y2
    tl = []
    for tid, g in broken.groupby("track_id"):
        idx = g.index.to_numpy()
        c = emb[idx].mean(0); c /= max(np.linalg.norm(c), 1e-9)
        g = g.sort_values("frame_idx")
        s0, s1 = int(g.frame_idx.iloc[0]), int(g.frame_idx.iloc[-1])
        start = np.array([g.fx.iloc[0], g.fy.iloc[0]]); end = np.array([g.fx.iloc[-1], g.fy.iloc[-1]])
        tl.append((tid, c, s0, s1, len(g), start, end))
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    return gt, broken, tl, rr


def motion_bonus(a, b, gmax, dmax):
    # a,b are tracklet tuples; order by time, measure end->start gap + distance
    (e_s0, e_s1, e_end), (l_s0, l_s1, l_start) = (a[2], a[3], a[6]), (b[2], b[3], b[5])
    if a[3] > b[2]:
        (e_s0, e_s1, e_end), (l_s0, l_s1, l_start) = (b[2], b[3], b[6]), (a[2], a[3], a[5])
    gap = l_s0 - e_s1
    if gap <= 0 or gap > gmax:
        return 0.0
    dist = float(np.hypot(*(e_end - l_start)))
    if dist > dmax:
        return 0.0
    return (1 - gap / gmax) * (1 - dist / dmax)


def relink(tl, rr, th, w, gmax=45, dmax=120):
    order = sorted(range(len(tl)), key=lambda i: tl[i][2] - tl[i][3])  # longest first
    clusters, assign = [], {}
    for i in order:
        f0, f1 = tl[i][2], tl[i][3]; best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < tl[m][2] or f0 > tl[m][3]) for m in cl):
                continue
            s = max(rr[i, m] + (w * motion_bonus(tl[i], tl[m], gmax, dmax) if w else 0.0) for m in cl)
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= th:
            clusters[best].append(i); assign[tl[i][0]] = best
        else:
            clusters.append([i]); assign[tl[i][0]] = len(clusters) - 1
    return assign, len(clusters)


def score(P, th, w):
    gt, broken, tl, rr = P
    a, k = relink(tl, rr, th, w)
    rel = broken.copy(); rel["track_id"] = broken.track_id.map(a)
    m = full_metrics(rel, gt); m["k"] = k
    return m


def main():
    Pt = prep(sys.argv[1]); Ph = prep(sys.argv[2])
    print(f"{'variant':<20}{'IDF1':>7}{'HOTA':>7}{'AssPr':>7}{'AssRe':>7}{'#k':>5}")
    for label, w in [("B1 (no motion)", 0.0), ("B5 w=0.10", 0.10), ("B5 w=0.20", 0.20), ("B5 w=0.35", 0.35)]:
        best = max((0.16, 0.20, 0.24, 0.28), key=lambda th: score(Pt, th, w)["IDF1"])
        mt = score(Pt, best, w); mh = score(Ph, best, w)
        print(f"{label + ' TUNE':<20}{mt['IDF1']:>7.3f}{mt['HOTA']:>7.3f}{mt['AssPr']:>7.3f}{mt['AssRe']:>7.3f}{mt['k']:>5d}", flush=True)
        print(f"{label + ' HELD':<20}{mh['IDF1']:>7.3f}{mh['HOTA']:>7.3f}{mh['AssPr']:>7.3f}{mh['AssRe']:>7.3f}{mh['k']:>5d}", flush=True)


if __name__ == "__main__":
    main()
