"""Decompose GS-DetA loss: for each GT person box, why is it not a true positive?
Categories: no-detection (no pred within 5 m), then among located preds which of
role / team / jersey breaks the IdSim gate. Reveals the dominant failure mode.

    python gsr/err_analysis.py <split> <seqs|all>
Reads the predictions written by s3b at out/preds/SoccerNetGS-<split>/ours/data/.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
import numpy as np

import common as C

PRED = C.OUT_ROOT / "preds"
SIGMA = 2.0427


def norm_team(t):
    return None if t in (None, "", "nan", "null") else t


def norm_jersey(j):
    if j in (None, "", "null", "nan"):
        return None
    try:
        return int(j)
    except (ValueError, TypeError):
        return None


def run(split, seqs):
    cat = Counter()
    by_role = {r: Counter() for r in ("player", "goalkeeper", "referee")}
    for seq in seqs:
        labels = C.load_labels(split, seq)
        pj = PRED / f"SoccerNetGS-{split}" / "ours" / "data" / f"{seq}.json"
        if not pj.exists():
            continue
        preds = json.load(open(pj))["predictions"]
        # GT person boxes per image_id
        gt = {}
        for a in labels["annotations"]:
            at = a.get("attributes") or {}
            if at.get("role") not in ("player", "goalkeeper", "referee"):
                continue
            bp = a.get("bbox_pitch")
            if not bp:
                continue
            gt.setdefault(a["image_id"], []).append(
                (bp["x_bottom_middle"], bp["y_bottom_middle"], at.get("role"),
                 norm_team(at.get("team")), norm_jersey(at.get("jersey"))))
        # preds per image_id
        pr = {}
        for p in preds:
            bp = p["bbox_pitch"]; at = p["attributes"]
            pr.setdefault(p["image_id"], []).append(
                (bp["x_bottom_middle"], bp["y_bottom_middle"], at.get("role"),
                 norm_team(at.get("team")), norm_jersey(at.get("jersey"))))
        for iid, gboxes in gt.items():
            pboxes = pr.get(iid, [])
            parr = np.array([[b[0], b[1]] for b in pboxes]) if pboxes else np.zeros((0, 2))
            for gx, gy, grole, gteam, gjer in gboxes:
                rc = by_role[grole]
                if len(parr):
                    d = np.hypot(parr[:, 0] - gx, parr[:, 1] - gy)
                    k = int(d.argmin())
                else:
                    d, k = None, None
                if k is None or d[k] > 5.0:
                    cat["no_detection"] += 1; rc["no_detection"] += 1
                    continue
                _, _, prole, pteam, pjer = pboxes[k]
                if prole != grole:
                    cat["role_wrong"] += 1; rc["role_wrong"] += 1
                elif grole in ("player", "goalkeeper") and pteam != gteam:
                    cat["team_wrong"] += 1; rc["team_wrong"] += 1
                elif grole == "player" and pjer != gjer:
                    cat["jersey_wrong"] += 1; rc["jersey_wrong"] += 1
                    rc["jersey_wrong_known" if gjer is not None else "jersey_wrong_none"] += 1
                else:
                    cat["TP"] += 1; rc["TP"] += 1
    tot = sum(cat.values())
    print(f"=== GT person boxes: {tot} ===")
    for k, v in cat.most_common():
        print(f"  {k:16s} {v:6d}  {100*v/max(1,tot):5.1f}%")
    for role, rc in by_role.items():
        t = sum(rc.values())
        if not t:
            continue
        print(f"\n[{role}] {t} GT boxes, TP-rate {100*rc['TP']/t:.1f}%")
        for k, v in rc.most_common():
            if k.startswith("jersey_wrong_") or k == "TP":
                continue
            print(f"    {k:16s} {v:6d}  {100*v/t:5.1f}%")
        if role == "player":
            print(f"    jersey_wrong: known={rc['jersey_wrong_known']} none={rc['jersey_wrong_none']}")


if __name__ == "__main__":
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    run(split, seqs)
