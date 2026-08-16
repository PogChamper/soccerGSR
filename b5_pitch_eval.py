"""B5 — pitch-coordinate reachability gate + bonus in the relink. Project each
tracklet's start/end foot point to PITCH METRES via the per-frame PnLCalib
homography (camera panning removed). A hard gate forbids merges implying >11 m/s
(protects AssPr); a graded bonus lowers the appearance bar for spatially close,
reachable transitions (raises AssRe). This is the calibrated fix for image-space
motion (B4) failing on the panning camera. Tune on tune, report on held-out."""
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
HALF_L, HALF_W = 57.5, 39.0


def project(H, cx, cy):
    w = H @ np.array([cx, cy, 1.0])
    if abs(w[2]) < 1e-9:
        return None
    x, y = w[0] / w[2], w[1] / w[2]
    return np.array([x, y]) if abs(x) <= HALF_L and abs(y) <= HALF_W else None


def tracklet_pitch(broken, tid_of, Hs, K=5, win=3):
    keys = np.array(sorted(Hs)); N = len(tid_of)
    start = np.full((N, 2), np.nan); end = np.full((N, 2), np.nan)
    ok_s = np.zeros(N, bool); ok_e = np.zeros(N, bool)
    groups = {t: g for t, g in broken.groupby("track_id")}

    def nearest(frame):
        j = int(np.searchsorted(keys, frame)); best = None
        for k in (keys[j - 1] if j > 0 else None, keys[j] if j < len(keys) else None):
            if k is not None and abs(k - frame) <= win and (best is None or abs(k - frame) < abs(best - frame)):
                best = k
        return Hs.get(int(best)) if best is not None else None

    for i, tid in enumerate(tid_of):
        g = groups[tid].sort_values("frame_idx")
        fr = g.frame_idx.to_numpy(); cx = ((g.x1 + g.x2) / 2).to_numpy(); cy = g.y2.to_numpy()

        def med(sl):
            pts = []
            for f, x, y in zip(fr[sl], cx[sl], cy[sl]):
                H = nearest(int(f))
                if H is not None:
                    p = project(H, x, y)
                    if p is not None:
                        pts.append(p)
            return np.median(pts, axis=0) if pts else None
        ps, pe = med(slice(0, K)), med(slice(max(0, len(fr) - K), len(fr)))
        if ps is not None:
            start[i], ok_s[i] = ps, True
        if pe is not None:
            end[i], ok_e[i] = pe, True
    return start, end, ok_s, ok_e


def greedy_pitch(order, sim, spans, thresh, R, vmax, beta, dfloor=3.0, vsoft=6.0, fps=30.0):
    clusters, assign = [], {}
    for i in order:
        f0, f1 = spans[i]; best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["spans"]):
                continue
            s = max(sim[i, m] for m in cl["members"])
            if R is not None:
                # nearest transition to a member; project reachability in metres
                m = min(cl["members"], key=lambda m: min(abs(spans[m][1] - f0), abs(f1 - spans[m][0])))
                if spans[m][1] <= f0:
                    dt, pp, okp, pn, okn = f0 - spans[m][1], R["end"][m], R["ok_e"][m], R["start"][i], R["ok_s"][i]
                else:
                    dt, pp, okp, pn, okn = spans[m][0] - f1, R["end"][i], R["ok_e"][i], R["start"][m], R["ok_s"][m]
                if okp and okn and dt > 0:
                    dist = float(np.linalg.norm(pp - pn)); secs = dt / fps
                    if dist > max(vmax * secs, dfloor):
                        continue  # unreachable -> hard gate
                    s = s + beta * (1.0 - min((dist / secs) / vsoft, 1.0))
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= thresh:
            clusters[best]["members"].append(i); clusters[best]["spans"].append((f0, f1)); assign[i] = best
        else:
            clusters.append(dict(members=[i], spans=[(f0, f1)])); assign[i] = len(clusters) - 1
    return assign, len(clusters)


def prep(cache_path, h_path):
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
    with open(h_path, "rb") as f:
        Hs = pickle.load(f)
    s0, e0, oks, oke = tracklet_pitch(broken, tid_of, Hs)
    R = dict(start=s0, end=e0, ok_s=oks, ok_e=oke)
    print(f"[{cache_path.split('/')[-1]}] tracklets {len(tl)}, pitch-valid endpoints {oks.mean():.2f}/{oke.mean():.2f}", flush=True)
    return gt, broken, spans, order, tid_of, rr, R


def sc(P, th, R, vmax, beta):
    gt, broken, spans, order, tid_of, rr, RR = P
    a, k = greedy_pitch(order, rr, spans, th, RR if R else None, vmax, beta)
    rel = broken.copy(); rel["track_id"] = broken.track_id.map({tid_of[i]: a[i] for i in a})
    m = full_metrics(rel, gt); m["k"] = k
    return m


def main():
    Pt = prep(sys.argv[1], sys.argv[3]); Ph = prep(sys.argv[2], sys.argv[4])
    print(f"{'variant':<22}{'IDF1':>7}{'HOTA':>7}{'AssPr':>7}{'AssRe':>7}{'#k':>5}")
    grid = (0.14, 0.18, 0.22, 0.26)
    cfgs = [("baseline", False, 11.0, 0.0), ("B5 gate", True, 11.0, 0.0),
            ("B5 gate+bonus", True, 11.0, 0.10), ("B5 gate+bonus lo-th", True, 11.0, 0.10)]
    lowgrid = (0.10, 0.13, 0.16, 0.20)
    for label, use_R, vmax, beta in cfgs:
        g = lowgrid if "lo-th" in label else grid
        th = max(g, key=lambda t: sc(Pt, t, use_R, vmax, beta)["IDF1"])
        mt = sc(Pt, th, use_R, vmax, beta); mh = sc(Ph, th, use_R, vmax, beta)
        print(f"{label + ' TUNE':<22}{mt['IDF1']:>7.3f}{mt['HOTA']:>7.3f}{mt['AssPr']:>7.3f}{mt['AssRe']:>7.3f}{mt['k']:>5d}", flush=True)
        print(f"{label + ' HELD':<22}{mh['IDF1']:>7.3f}{mh['HOTA']:>7.3f}{mh['AssPr']:>7.3f}{mh['AssRe']:>7.3f}{mh['k']:>5d}", flush=True)


if __name__ == "__main__":
    main()
