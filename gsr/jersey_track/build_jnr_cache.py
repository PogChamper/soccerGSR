"""D5 GS-HOTA leg: write generic (number, conf) jersey caches for valid-12 from the JNR posteriors.

Rules (per detection box):
  jnr      : number = JNR argmax where Dirichlet uncertainty <= u_th, conf = w_j; else -1
  jnrmix   : as jnr, else the ConvNeXt read where VIS >= 0.7 and CONF >= 0.95, conf = 1.0; else -1
The s3b generic vote counts conf-weighted numbers over boxes with conf >= GSR_OCR_CONF_TH, so run the
chain with GSR_OCR_CONF_TH=0.5 (all encoded votes admitted) and GSR_JERSEY_FMT=generic.
Boxes without a JNR read (referees, unscored) fall back to the ConvNeXt rule in jnrmix and to -1 in jnr.

    python gsr/jersey_track/build_jnr_cache.py --u 0.05 --w 2 --tag jnrmix_u05w2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")

import numpy as np
from collections import Counter
import pandas as pd

import common as C


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/d/jersey-lab/jnr_valid12")
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["jnr", "jnrmix", "cnxjgate", "consensus", "jnrstar"], default="jnrmix",
                    help="cnxjgate: ConvNeXt read where JNR uncertainty <= u (JNR as the visibility gate) and CONF >= 0.95")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--second", default="ctrl_old_devval", help="consensus: tag of the second ConvNeXt cache (jersey_<tag>.pkl)")
    ap.add_argument("--conf2", type=float, default=0.9, help="consensus: CONF gate of the second reader")
    ap.add_argument("--star", default="star_v1_nogate", help="jnrstar: generic star cache tag used where JNR is uncertain")
    ap.add_argument("--star-conf", type=float, default=0.5, help="jnrstar: min conf of the star read")
    args = ap.parse_args()
    d = Path(args.dir)
    seqs = sorted({p.stem.removesuffix("_unmatched") for p in d.glob("SNGS-*.npz")})
    for seq in seqs:
        det = C.load(C.OUT_ROOT / args.split / seq / "det.pkl")
        jer = C.load(C.OUT_ROOT / args.split / seq / "jersey.pkl")["frames"]
        jer2 = C.load(C.OUT_ROOT / args.split / seq / f"jersey_{args.second}.pkl")["frames"] if args.mode == "consensus" else None
        star = C.load(C.OUT_ROOT / args.split / seq / f"jersey_{args.star}.pkl")["frames"] if args.mode == "jnrstar" else None
        parts = []
        for sfx in ("", "_unmatched"):
            csv = d / f"{seq}{sfx}.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                z = np.load(d / f"{seq}{sfx}.npz")
                df["jnr_pred"] = z["probs"].astype(np.float32).argmax(1)
                df["jnr_unc"] = z["uncertainty"]
                parts.append(df[["frame_idx", "det_idx", "jnr_pred", "jnr_unc"]])
        jn = pd.concat(parts).set_index(["frame_idx", "det_idx"])
        frames = []
        n_j = n_c = 0
        for fi, boxes in enumerate(det["frames"]):
            n = len(boxes)
            nums, confs = np.full(n, -1, np.int32), np.zeros(n, np.float32)
            vis, tens, units = jer[fi]
            for j in range(n):
                key = (fi, j)
                if args.mode == "consensus":
                    # three readers per box; a box votes only when >= 2 admitted readers agree
                    reads = []
                    if key in jn.index and jn.at[key, "jnr_unc"] <= args.u:
                        reads.append(int(jn.at[key, "jnr_pred"]))
                    if vis[j] > -1e8:
                        v = 1 / (1 + np.exp(-vis[j]))
                        pt, pu = softmax(tens[j]), softmax(units[j])
                        if v >= 0.7 and min(pt.max(), pu.max()) >= 0.95:
                            t_, u_ = int(pt.argmax()), int(pu.argmax()); reads.append(u_ if t_ == 0 else t_ * 10 + u_)
                        v2, t2, u2 = jer2[fi]
                        pt2, pu2 = softmax(t2[j]), softmax(u2[j])
                        if v >= 0.7 and min(pt2.max(), pu2.max()) >= args.conf2:
                            t_, u_ = int(pt2.argmax()), int(pu2.argmax()); reads.append(u_ if t_ == 0 else t_ * 10 + u_)
                    if len(reads) >= 2:
                        c = Counter(reads).most_common(1)[0]
                        if c[1] >= 2:
                            nums[j], confs[j] = c[0], float(c[1])
                            n_c += 1
                    continue
                if args.mode == "jnrstar":
                    if key in jn.index and jn.at[key, "jnr_unc"] <= args.u:
                        nums[j], confs[j] = int(jn.at[key, "jnr_pred"]), args.w
                        n_j += 1
                    else:
                        sn, sc = star[fi]
                        if j < len(sn) and sn[j] >= 0 and sc[j] >= args.star_conf:
                            nums[j], confs[j] = int(sn[j]), 1.0
                            n_c += 1
                    continue
                if args.mode == "cnxjgate":
                    if key in jn.index and jn.at[key, "jnr_unc"] <= args.u and vis[j] > -1e8:
                        pt, pu = softmax(tens[j]), softmax(units[j])
                        if min(pt.max(), pu.max()) >= 0.95:
                            t_, u_ = int(pt.argmax()), int(pu.argmax())
                            nums[j], confs[j] = (u_ if t_ == 0 else t_ * 10 + u_), 1.0
                            n_c += 1
                    continue
                if key in jn.index and jn.at[key, "jnr_unc"] <= args.u:
                    nums[j], confs[j] = int(jn.at[key, "jnr_pred"]), args.w
                    n_j += 1
                elif args.mode == "jnrmix" and vis[j] > -1e8:
                    v = 1 / (1 + np.exp(-vis[j]))
                    pt, pu = softmax(tens[j]), softmax(units[j])
                    if v >= 0.7 and min(pt.max(), pu.max()) >= 0.95:
                        t, u = int(pt.argmax()), int(pu.argmax())
                        nums[j], confs[j] = (u if t == 0 else t * 10 + u), 1.0
                        n_c += 1
            frames.append((nums, confs))
        C.dump({"frames": frames}, C.OUT_ROOT / args.split / seq / f"jersey_{args.tag}.pkl")
        print(f"[cache {seq}] jersey_{args.tag}.pkl: {n_j} JNR votes, {n_c} ConvNeXt votes", flush=True)


if __name__ == "__main__":
    main()
