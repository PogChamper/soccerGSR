"""Compact HOTA / AssA / AssPr / AssRe (plus DetA) for the fast-test harness.
py-motmetrics gives IDF1/MOTA but not the association axis; MOTA 0.91 + IDF1 0.47
is the textbook detection-solved / association-limited fingerprint, and our failure
is low AssPr (teammate MERGES). This exposes that axis.

Follows the TrackEval HOTA formulation with a per-frame IoU (Hungarian) matching
approximation, averaged over localisation thresholds alpha. Good enough to steer
A/B decisions on merges; not a leaderboard submission."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum.outer(a[:, 0], b[:, 0]); y1 = np.maximum.outer(a[:, 1], b[:, 1])
    x2 = np.minimum.outer(a[:, 2], b[:, 2]); y2 = np.minimum.outer(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def hota(gt, hyp, alphas=np.array([0.3, 0.5, 0.7])):
    """gt: frame,id,x1..y2 ; hyp: frame,track_id,x1..y2. Returns dict of metrics
    (HOTA/DetA/AssA/AssPr/AssRe averaged over alpha)."""
    frames = sorted(set(gt.frame.unique()) | set(hyp.frame.unique()))
    gt_g = {f: g for f, g in gt.groupby("frame")}
    hy_g = {f: g for f, g in hyp.groupby("frame")}
    gcount = gt.id.value_counts().to_dict()
    hcount = hyp.track_id.value_counts().to_dict()

    out = {k: [] for k in ("HOTA", "DetA", "AssA", "AssPr", "AssRe")}
    for al in alphas:
        M = {}  # (gid,hid) -> co-matched frame count
        TP = FP = FN = 0
        for f in frames:
            gf = gt_g.get(f); hf = hy_g.get(f)
            ng = 0 if gf is None else len(gf)
            nh = 0 if hf is None else len(hf)
            if ng == 0:
                FP += nh; continue
            if nh == 0:
                FN += ng; continue
            iou = _iou_matrix(gf[["x1", "y1", "x2", "y2"]].to_numpy(float),
                              hf[["x1", "y1", "x2", "y2"]].to_numpy(float))
            ri, ci = linear_sum_assignment(-iou)
            matched_g = set(); matched_h = set()
            gids = gf.id.to_numpy(); hids = hf.track_id.to_numpy()
            for r, c in zip(ri, ci):
                if iou[r, c] >= al:
                    TP += 1; matched_g.add(r); matched_h.add(c)
                    key = (int(gids[r]), int(hids[c]))
                    M[key] = M.get(key, 0) + 1
            FP += nh - len(matched_h); FN += ng - len(matched_g)
        if TP == 0:
            for k in out:
                out[k].append(0.0)
            continue
        deta = TP / (TP + FP + FN)
        a_sum = pr_sum = re_sum = 0.0
        for (g, h), tpa in M.items():
            fna = gcount[g] - tpa; fpa = hcount[h] - tpa
            a_sum += tpa * (tpa / (tpa + fna + fpa))
            pr_sum += tpa * (tpa / (tpa + fpa))
            re_sum += tpa * (tpa / (tpa + fna))
        assa = a_sum / TP; asspr = pr_sum / TP; assre = re_sum / TP
        out["DetA"].append(deta); out["AssA"].append(assa)
        out["AssPr"].append(asspr); out["AssRe"].append(assre)
        out["HOTA"].append(np.sqrt(deta * assa))
    return {k: float(np.mean(v)) for k, v in out.items()}
