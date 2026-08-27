"""Perfect-association jersey diagnostic: group detector crops by GT identity
(best IoU >= 0.5), replay the exact s3b vote, and split the failures.

Reproduces the LOG.md:70 '82.4% per-GT-track, only 9 misses' claim with an artifact
and adds the crop-supply breakdown behind every abstention.

Both jersey cache formats are replayed, picked by the arity of a frame entry as the
assembler picks GSR_JERSEY_FMT: 3-tuples (vis, tens, units) are the ConvNeXt two-head
logits, 2-tuples (numbers, confidences) a generic reader. VIS_TH is unused by the
generic branch (no visibility head), CONF_TH gates the read confidence in both.

    python oracle_assoc.py <split> <seq,seq,...> [VIS_TH] [CONF_TH] [MIN_VOTES]
Env: GSR_JERSEY_TAG ("" -> jersey.pkl, else jersey_<tag>.pkl).
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GT_ROOT = Path("/mnt/d/datasets/soccernet2025")
OUT = Path("/home/dxdxxd/projects/soccer-app/gsr/out")
VIS_TH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
CONF_TH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
MIN_VOTES = int(sys.argv[5]) if len(sys.argv) > 5 else 6


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def sm(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def iou_to(gb, box):
    ix1 = np.maximum(gb[:, 0], box[0]); iy1 = np.maximum(gb[:, 1], box[1])
    ix2 = np.minimum(gb[:, 2], box[2]); iy2 = np.minimum(gb[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def run(split, seqs):
    tot = Counter()
    miss_detail = Counter()
    generic = False
    for seq in seqs:
        lab = json.load(open(GT_ROOT / split / seq / "Labels-GameState.json"))
        imgs = sorted(lab["images"], key=lambda im: im["image_id"])
        pos = {im["image_id"]: i for i, im in enumerate(imgs)}
        gt = defaultdict(lambda: ([], []))
        gtj = defaultdict(list)
        for a in lab["annotations"]:
            at = a.get("attributes") or {}
            if at.get("role") != "player" or a["image_id"] not in pos:
                continue
            b = a["bbox_image"]
            g = gt[pos[a["image_id"]]]
            g[0].append((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
            g[1].append(a["track_id"])
            gtj[a["track_id"]].append(at.get("jersey"))
        gtnum = {}
        for tid, js in gtj.items():
            k = [j for j in js if j not in (None, "", "null")]
            gtnum[tid] = Counter(k).most_common(1)[0][0] if k else None

        jtag = os.environ.get("GSR_JERSEY_TAG", "")
        jer = pickle_load(OUT / split / seq / (f"jersey_{jtag}.pkl" if jtag else "jersey.pkl"))["frames"]
        tk = pickle_load(OUT / split / seq / "track_base.pkl")["records"]
        generic = len(jer[0]) == 2
        # votes, nv (accepted reads), nvis (crops past the visibility gate), ncrop
        acc = defaultdict(lambda: [Counter() if generic else [np.zeros(10), np.zeros(10)], 0, 0, 0])
        for fi, j, _tid, cls, bbox, _c in tk:
            if cls != 0:
                continue
            g = gt.get(fi)
            if not g or not len(g[0]):
                continue
            gb = np.array(g[0], float)
            ious = iou_to(gb, bbox)
            k = int(ious.argmax())
            if ious[k] < 0.5:
                continue
            gid = g[1][k]
            a = acc[gid]
            a[3] += 1
            if generic:
                nums, confs = jer[fi]
                if j >= len(nums):
                    continue
                a[2] += 1
                n, c = int(nums[j]), float(confs[j])
                if n < 0 or c < CONF_TH:
                    continue
                a[0][n] += c; a[1] += 1
                continue
            vis, tens, units = jer[fi]
            if j >= len(vis):
                continue
            if sig(vis[j]) < VIS_TH:
                continue
            a[2] += 1
            if min(sm(tens[j]).max(), sm(units[j]).max()) < CONF_TH:
                continue
            a[0][0] += tens[j]; a[0][1] += units[j]; a[1] += 1

        for gid, gnum in gtnum.items():
            a = acc.get(gid)
            if a is None:
                tot["gt_track_no_detection"] += 1
                continue
            votes, nv, nvis, ncrop = a
            ours = None
            if nv >= MIN_VOTES:
                if generic:
                    ours = int(max(votes, key=votes.get))
                else:
                    t, u = int(np.argmax(votes[0])), int(np.argmax(votes[1]))
                    ours = u if t == 0 else t * 10 + u
            if gnum is None:
                tot["Nnone_commit" if ours is not None else "Nnone_ok"] += 1
                continue
            if ours is None:
                tot["known_miss"] += 1
                if ncrop < MIN_VOTES:
                    miss_detail["crops<MV (no supply)"] += 1
                elif nvis < MIN_VOTES:
                    miss_detail["gate killed it (vis<MV)"] += 1
                else:
                    miss_detail["conf killed it (vis>=MV, nv<MV)"] += 1
            elif str(ours) == str(gnum):
                tot["known_hit"] += 1
            else:
                tot["known_wrong"] += 1
    kn = tot["known_hit"] + tot["known_miss"] + tot["known_wrong"]
    fmt = "generic" if generic else "logits"
    tag = os.environ.get("GSR_JERSEY_TAG", "") or "jersey.pkl"
    print(f"fmt={fmt} tag={tag} VIS={VIS_TH if not generic else '-'} "
          f"CONF={CONF_TH} MIN_VOTES={MIN_VOTES}  seqs={len(seqs)}")
    print(f"GT known-jersey player tracks evaluated: {kn}")
    print(f"  hit  {tot['known_hit']:4d}  ({100*tot['known_hit']/max(1,kn):.1f}%)")
    print(f"  wrong{tot['known_wrong']:4d}  ({100*tot['known_wrong']/max(1,kn):.1f}%)")
    print(f"  miss {tot['known_miss']:4d}  ({100*tot['known_miss']/max(1,kn):.1f}%)")
    print(f"  GT None-track: correct-abstain={tot['Nnone_ok']} false-commit={tot['Nnone_commit']}")
    print(f"  GT tracks with no matched detection at all: {tot['gt_track_no_detection']}")
    print("  miss breakdown:", dict(miss_detail))


def pickle_load(p):
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(","))
