"""B4 — appearance-free motion reachability GATE in the relink. The AssRe headroom
is short-gap same-kit fragments (79% gap<1s) that appearance can't clear. A hard
kinematic gate (constant-velocity foot residual, body-height normalized) lets us
LOWER the appearance threshold safely: motion vetoes the wrong merges the lower
threshold would let in, while the true short-gap links now clear. Targets AssRe."""
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


def kinematics(broken, k=8):
    K = {}
    for tid, g in broken.groupby("track_id"):
        g = g.sort_values("frame_idx")
        fr = g.frame_idx.to_numpy()
        fx = ((g.x1 + g.x2) / 2).to_numpy(); fy = g.y2.to_numpy(); h = (g.y2 - g.y1).to_numpy()
        kk = min(k, len(g))
        dt_t = max(fr[-1] - fr[-kk], 1); dt_h = max(fr[kk - 1] - fr[0], 1)
        K[tid] = dict(f0=int(fr[0]), f1=int(fr[-1]), x0=fx[0], y0=fy[0], x1=fx[-1], y1=fy[-1],
                      vtx=(fx[-1] - fx[-kk]) / dt_t, vty=(fy[-1] - fy[-kk]) / dt_t,
                      vhx=(fx[kk - 1] - fx[0]) / dt_h, vhy=(fy[kk - 1] - fy[0]) / dt_h,
                      h=float(np.median(h)))
    return K


def reach_ok(A, B, r_max=1.5, v_max=0.06):
    """Is A->B kinematically plausible? A precedes B. Body-height-normalized CV residual."""
    gap = B["f0"] - A["f1"]
    if gap <= 0:
        return False
    H = max(0.5 * (A["h"] + B["h"]), 1e-6)
    pfx, pfy = A["x1"] + A["vtx"] * gap, A["y1"] + A["vty"] * gap
    fwd_res = np.hypot(pfx - B["x0"], pfy - B["y0"]) / H
    return fwd_res <= r_max + v_max * gap


def greedy_gate(order, sim, spans, thresh, K, tids, use_gate):
    clusters, assign = [], {}
    for i in order:
        f0, f1 = spans[i]; Ki = K[tids[i]]; best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["spans"]):
                continue
            if use_gate:
                m = min(cl["members"], key=lambda m: min(abs(spans[m][1] - f0), abs(f1 - spans[m][0])))
                A, B = (K[tids[m]], Ki) if spans[m][1] <= f0 else (Ki, K[tids[m]])
                if not reach_ok(A, B):
                    continue
            s = max(sim[i, m] for m in cl["members"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= thresh:
            clusters[best]["members"].append(i); clusters[best]["spans"].append((f0, f1)); assign[tids[i]] = best
        else:
            clusters.append(dict(members=[i], spans=[(f0, f1)])); assign[tids[i]] = len(clusters) - 1
    return assign, len(clusters)


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
    tl = []
    for tid, g in broken.groupby("track_id"):
        idx = g.index.to_numpy(); c = emb[idx].mean(0); c /= max(np.linalg.norm(c), 1e-9)
        tl.append((tid, c, int(g.frame_idx.min()), int(g.frame_idx.max())))
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]; tid_of = [t[0] for t in tl]
    order = sorted(range(len(tl)), key=lambda i: spans[i][0] - spans[i][1])
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    K = kinematics(broken)
    return gt, broken, spans, order, tid_of, rr, K


def sc(P, th, use_gate):
    gt, broken, spans, order, tid_of, rr, K = P
    a, k = greedy_gate(order, rr, spans, th, K, tid_of, use_gate)
    rel = broken.copy(); rel["track_id"] = broken.track_id.map({tid_of[i]: a[i] for i in a})
    m = full_metrics(rel, gt); m["k"] = k
    return m


def main():
    Pt = prep(sys.argv[1]); Ph = prep(sys.argv[2])
    print(f"{'variant':<24}{'IDF1':>7}{'HOTA':>7}{'AssPr':>7}{'AssRe':>7}{'#k':>5}")
    # baseline (no gate) tune-optimal, then gate at same + lowered thresholds
    grid = (0.14, 0.18, 0.22, 0.26, 0.30)
    base_th = max(grid, key=lambda th: sc(Pt, th, False)["IDF1"])
    for label, use_gate, ths in [("baseline", False, [base_th]),
                                 ("+gate same-th", True, [base_th]),
                                 ("+gate lower-th", True, [0.10, 0.13, 0.16])]:
        if label == "+gate lower-th":
            th = max(ths, key=lambda t: sc(Pt, t, True)["IDF1"])
        else:
            th = ths[0]
        mt = sc(Pt, th, use_gate); mh = sc(Ph, th, use_gate)
        print(f"{label + ' TUNE':<24}{mt['IDF1']:>7.3f}{mt['HOTA']:>7.3f}{mt['AssPr']:>7.3f}{mt['AssRe']:>7.3f}{mt['k']:>5d}", flush=True)
        print(f"{label + ' HELD':<24}{mh['IDF1']:>7.3f}{mh['HOTA']:>7.3f}{mh['AssPr']:>7.3f}{mh['AssRe']:>7.3f}{mh['k']:>5d}", flush=True)


if __name__ == "__main__":
    main()
