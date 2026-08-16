"""Phase-1 evaluator: offline relink (cosine vs k-reciprocal) scored with the full
metric set — IDF1/MOTA/recall (py-motmetrics) + HOTA/AssA/AssPr/AssRe (hota.py) —
on a given cached segment. Run on the tune segment and the held-out segment; tune
the merge threshold on tune, report on held-out."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np

if not hasattr(np, "asfarray"):  # motmetrics 1.4 vs numpy 2.0
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd

from hota import hota
from lever1_break import occlusion_break
from lever5_relink import attach_emb
from lever_b1_rerank import greedy, tracklets
from track_bench import GT, replay
from ltpi_research.advanced import k_reciprocal_rerank


def full_metrics(rel, gt):
    acc = mm.MOTAccumulator(auto_id=False)
    rel = rel.copy(); rel["image_id"] = rel.frame_idx + 1
    by = {f: g for f, g in rel.groupby("image_id")}
    for img, gtf in gt.groupby("image_id"):
        h = by.get(img)
        gb = np.c_[gtf.x1, gtf.y1, gtf.x2 - gtf.x1, gtf.y2 - gtf.y1]
        hb = np.c_[h.x1, h.y1, h.x2 - h.x1, h.y2 - h.y1] if h is not None else np.zeros((0, 4))
        d = mm.distances.iou_matrix(gb, hb, max_iou=0.5)
        acc.update(gtf.id.tolist(), (h.track_id.tolist() if h is not None else []), d, frameid=img)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["idf1", "mota", "recall", "num_switches"], name="x").iloc[0]
    ht = hota(gt.rename(columns={"image_id": "frame"}),
              rel.rename(columns={"image_id": "frame"}))
    return dict(IDF1=s.idf1, HOTA=ht["HOTA"], AssA=ht["AssA"], AssPr=ht["AssPr"],
                AssRe=ht["AssRe"], recall=s.recall, MOTA=s.mota, IDsw=int(s.num_switches))


def prep(cache_path, video):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT); gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)].rename(columns={"image_id": "image_id"})
    gt = gt.rename(columns={"id": "id"})
    gt["image_id"] = gt.image_id  # keep
    obs = replay(cache, video, dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90))
    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs).reset_index(drop=True)
    emb = attach_emb(obs, cache)[broken.orig.to_numpy()]
    tl = tracklets(broken, emb)
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]
    order = sorted(range(len(tl)), key=lambda i: -tl[i][4])
    tid_of = [t[0] for t in tl]
    cos = cents @ cents.T
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    return dict(gt=gt, broken=broken, spans=spans, order=order, tid_of=tid_of, cos=cos, rr=rr)


def evalsim(P, sim, th):
    a, k = greedy(P["order"], sim, P["spans"], th)
    rel = P["broken"].copy()
    rel["track_id"] = P["broken"].track_id.map({P["tid_of"][i]: a[i] for i in a})
    m = full_metrics(rel, P["gt"]); m["k"] = k
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    Pt = prep(args.tune, args.video)
    Ph = prep(args.held, args.video)

    hdr = f"{'method/seg':<20}{'th':>5}{'IDF1':>7}{'HOTA':>7}{'AssA':>7}{'AssPr':>7}{'AssRe':>7}{'rec':>6}{'MOTA':>6}{'#k':>5}"
    grids = {"cosine": (0.55, 0.62, 0.68, 0.74, 0.80), "k-recip": (0.20, 0.28, 0.34, 0.42, 0.50)}
    for meth, grid in grids.items():
        print("\n" + hdr)
        best = None
        for th in grid:
            m = evalsim(Pt, Pt[meth == "cosine" and "cos" or "rr"], th)
            print(f"{'%s TUNE' % meth:<20}{th:>5.2f}{m['IDF1']:>7.3f}{m['HOTA']:>7.3f}{m['AssA']:>7.3f}{m['AssPr']:>7.3f}{m['AssRe']:>7.3f}{m['recall']:>6.2f}{m['MOTA']:>6.2f}{m['k']:>5d}", flush=True)
            if best is None or m["IDF1"] > best[1]:
                best = (th, m["IDF1"])
        th = best[0]
        mh = evalsim(Ph, Ph[meth == "cosine" and "cos" or "rr"], th)
        print(f"{'%s HELD@tuned' % meth:<20}{th:>5.2f}{mh['IDF1']:>7.3f}{mh['HOTA']:>7.3f}{mh['AssA']:>7.3f}{mh['AssPr']:>7.3f}{mh['AssRe']:>7.3f}{mh['recall']:>6.2f}{mh['MOTA']:>6.2f}{mh['k']:>5d}", flush=True)


if __name__ == "__main__":
    main()
