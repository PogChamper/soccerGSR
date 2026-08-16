"""Diagnose per-track jersey decisions against GT (soccer_eda env).

Maps each BoT-SORT track to its GT track (majority IoU>=0.5), derives the GT
per-track number (majority of non-None labels, else None), recomputes our voted
jersey, and buckets the outcome so we can see where the DetA gap to the oracle
comes from: false commits on None-tracks vs misses/wrong-reads on known-tracks.

    python gsr/diag_jersey.py <split> <seqs|all>
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
import numpy as np

import common as C

VIS_TH = float(os.environ.get("GSR_VIS_TH", "0.6"))
OCR_CONF_TH = float(os.environ.get("GSR_OCR_CONF_TH", "0.7"))
MIN_VOTES = int(os.environ.get("GSR_MIN_VOTES", "4"))
TRACK_TAG = os.environ.get("GSR_TRACK_TAG", "base")


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def sm(z):
    e = np.exp(z - z.max()); return e / e.sum()


def iou_to(gb, box):
    ix1 = np.maximum(gb[:, 0], box[0]); iy1 = np.maximum(gb[:, 1], box[1])
    ix2 = np.minimum(gb[:, 2], box[2]); iy2 = np.minimum(gb[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]); ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def run(split, seqs):
    tot = Counter()
    for seq in seqs:
        base = C.OUT_ROOT / split / seq
        jer = C.load(base / "jersey.pkl")["frames"]
        tk = C.load(base / f"track_{TRACK_TAG}.pkl")["records"]
        labels = C.load_labels(split, seq)
        image_ids = C.frame_index(labels)[0]
        pos = {iid: fi for fi, iid in enumerate(image_ids)}
        gt = defaultdict(lambda: ([], [], []))  # fi -> boxes, gttid, jersey
        for a in labels["annotations"]:
            at = a.get("attributes") or {}
            if at.get("role") != "player" or a["image_id"] not in pos:
                continue
            b = a["bbox_image"]
            g = gt[pos[a["image_id"]]]
            g[0].append((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
            g[1].append(a["track_id"]); g[2].append(at.get("jersey"))

        by_track = defaultdict(list)
        for r in tk:
            by_track[r[2]].append(r)
        # GT per-track number
        gtnum = {}
        for gtid_boxes in [1]:
            pass
        gtjers = defaultdict(list)
        for fi, (boxes, gtids, jers) in gt.items():
            for gid, jn in zip(gtids, jers):
                gtjers[gid].append(jn)
        gt_track_num = {}
        for gid, js in gtjers.items():
            known = [j for j in js if j not in (None, "", "null")]
            gt_track_num[gid] = Counter(known).most_common(1)[0][0] if known else None

        for tid, rs in by_track.items():
            # our vote
            st, su, nv = np.zeros(10), np.zeros(10), 0
            gmatch = Counter()
            for fi, j, _, cls, bbox, _ in rs:
                if cls != 0:
                    continue
                g = gt.get(fi)
                if g and len(g[0]):
                    gb = np.array(g[0], float); k = int(iou_to(gb, bbox).argmax())
                    if iou_to(gb, bbox)[k] >= 0.5:
                        gmatch[g[1][k]] += 1
                vis, tens, units = jer[fi]
                if j >= len(vis) or sig(vis[j]) < VIS_TH:
                    continue
                if min(sm(tens[j]).max(), sm(units[j]).max()) < OCR_CONF_TH:
                    continue
                st += tens[j]; su += units[j]; nv += 1
            ours = None
            if nv >= MIN_VOTES:
                t, u = int(np.argmax(st)), int(np.argmax(su)); ours = str(u if t == 0 else t * 10 + u)
            if not gmatch:
                continue
            gid = gmatch.most_common(1)[0][0]
            gnum = gt_track_num.get(gid)
            if gnum is None:
                tot["Nnone_commit" if ours is not None else "Nnone_ok"] += 1
            else:
                if ours is None:
                    tot["known_miss"] += 1
                elif ours == gnum:
                    tot["known_hit"] += 1
                else:
                    tot["known_wrong"] += 1
    print(f"tracks matched to GT: {sum(tot.values())}")
    print(f"  GT None-track: correct-abstain={tot['Nnone_ok']}  FALSE-commit={tot['Nnone_commit']}")
    print(f"  GT known-track: hit={tot['known_hit']}  miss(abstain)={tot['known_miss']}  wrong={tot['known_wrong']}")
    kn = tot['known_hit'] + tot['known_miss'] + tot['known_wrong']
    print(f"  known-track accuracy: {tot['known_hit']}/{kn} = {tot['known_hit']/max(1,kn):.3f}")


if __name__ == "__main__":
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    run(split, seqs)
