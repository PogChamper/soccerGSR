"""Fragment rail: score tracklet-model commit rules on REAL DBSCAN fragments.

Runs on any split that has det/emb/track_base caches (built for dev-40 so the
operating point is selected on the population the chain actually contains:
fragments, ghosts, box masses). GT matching is in image space (no calibration).
Objective mirrors the measured valid post-mortem: a right commit on a GT-known
fragment earns its box mass, a commit on an alive GT-None fragment loses its
mass, wrong commits on known fragments are free (evaluator asymmetry), commits
on unmatched (ghost) fragments are free.

    python frag_rail.py infer --split valid --seqs <dev40> --ckpt .../best.pt
    python frag_rail.py select --dump /mnt/d/jersey-lab/tracklets_v1/frag_rail_dev40.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr/jersey_track")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

import common as C
from build_bt3_top4_cache import split_tracks, load_pickle
from tracklet_pix import TrackletPix, EVAL_TF, SIZE

DEV40 = ",".join(f"SNGS-{i:03d}" for i in list(range(39, 60)) + list(range(78, 97)))
DUMP = Path("/mnt/d/jersey-lab/tracklets_v1/frag_rail_dev40.csv")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_index(split: str, seq: str):
    d = json.load(open(f"/mnt/d/datasets/soccernet2025/{split}/{seq}/Labels-GameState.json"))
    by = defaultdict(list)
    for a in d["annotations"]:
        if a.get("category_id") in (1, 2, 3):
            bb = a["bbox_image"]
            by[int(str(a["image_id"])[-4:])].append(
                (bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"],
                 a["attributes"].get("jersey"), a.get("category_id")))
    return by


def pick_crops(mem: list, k: int = 16, gap: int = 8):
    """Top-k member boxes by area*conf^2 with a temporal gap, time-ordered."""
    scored = sorted(mem, key=lambda m: -(m[3] * max(m[4], 0.05) ** 2))
    chosen: list = []
    for m in scored:
        if len(chosen) >= k:
            break
        if all(abs(m[0] - c[0]) >= gap for c in chosen):
            chosen.append(m)
    for m in scored:
        if len(chosen) >= k:
            break
        if m not in chosen:
            chosen.append(m)
    return sorted(chosen, key=lambda m: m[0])


def cmd_infer(args):
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = TrackletPix().to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    rows = []
    for seq in args.seqs.split(","):
        sd = C.OUT_ROOT / args.split / seq
        track = load_pickle(sd / "track_base.pkl")["records"]
        emb = load_pickle(sd / "emb.pkl")["frames"]
        records = split_tracks(track, emb, 0.22)
        gt = gt_index(args.split, seq)
        frames_dir = C.seq_dir(args.split, seq) / "img1"
        mem_of = defaultdict(list)
        for fi, j, tid, cls, bbox, conf in records:
            x1, y1, x2, y2 = bbox
            mem_of[int(tid)].append((int(fi), int(j), list(map(float, bbox)),
                                     (x2 - x1) * (y2 - y1), float(conf)))
        cache: dict[int, np.ndarray] = {}
        for tid, mem in mem_of.items():
            votes, matched = Counter(), 0
            for fi, j, bbox, *_ in mem:
                best, bj = 0.0, None
                for g in gt.get(fi + 1, []):
                    v = iou(bbox, g[:4])
                    if v > best:
                        best, bj = v, g[4]
                if best >= 0.5:
                    matched += 1
                    votes[bj] += 1
            crops = []
            for fi, j, bbox, *_ in pick_crops(mem):
                if fi not in cache:
                    cache[fi] = cv2.imread(str(frames_dir / f"{fi + 1:06d}.jpg"))
                fr = cache[fi]
                h, w = fr.shape[:2]
                x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                                  int(min(w, bbox[2])), int(min(h, bbox[3])))
                if x2 - x1 < 4 or y2 - y1 < 8:
                    continue
                crops.append(EVAL_TF(image=cv2.cvtColor(fr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))["image"])
            if len(cache) > 64:
                cache.clear()
            if not crops:
                continue
            x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2)[None].to(args.device)
            mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=args.device)
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=args.device == "cuda"):
                t, u, a = model(x, mask)[:3]
            pt, pu = torch.softmax(t.float(), 1)[0], torch.softmax(u.float(), 1)[0]
            num = int(pu.argmax()) if int(pt.argmax()) == 0 else int(pt.argmax()) * 10 + int(pu.argmax())
            top = votes.most_common(1)
            gt_lab = top[0][0] if top else "UNMATCHED"
            rows.append(dict(seq=seq, frag=tid, n_boxes=len(mem), matched=matched,
                             gt=gt_lab if gt_lab is not None else "NONE",
                             gt_frac=round(top[0][1] / max(matched, 1), 3) if top else 0.0,
                             num=num, conf=round(float(pt.max() * pu.max()), 4),
                             absent=round(float(torch.sigmoid(a.float())[0]), 4)))
        print(f"[{seq}] {len(mem_of)} fragments", flush=True)
    import csv
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    with open(DUMP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {DUMP}: {len(rows)} fragments")


def cmd_select(args):
    import pandas as pd
    d = pd.read_csv(args.dump)
    known = d[(d["gt"] != "NONE") & (d["gt"] != "UNMATCHED")]
    alive_none = d[(d["gt"] == "NONE") & (d.matched >= 0.5 * d.n_boxes)]
    print(f"fragments: {len(d)} total, known {len(known)}, alive GT-None {len(alive_none)}, "
          f"ghost {int((d['gt'] == 'UNMATCHED').sum())}")
    best = None
    for tau in np.arange(0.05, 0.65, 0.05):
        for c in np.arange(0.4, 0.99, 0.02):
            kc = known[(known.absent <= tau) & (known.conf >= c)]
            gain = int(kc[kc.num.astype(str) == kc["gt"].astype(str)].n_boxes.sum())
            nc = alive_none[(alive_none.absent <= tau) & (alive_none.conf >= c)]
            loss = int(nc.n_boxes.sum())
            score = gain - args.lam * loss
            if best is None or score > best[0]:
                best = (score, round(tau, 2), round(c, 2), gain, loss,
                        int((kc.num.astype(str) == kc["gt"].astype(str)).sum()),
                        int((kc.num.astype(str) != kc["gt"].astype(str)).sum()), len(nc))
    print("select (lambda %.1f): score %d at tau %.2f conf %.2f -> "
          "gain %d boxes, loss %d boxes, right %d wrong %d alive-none-commits %d" % ((args.lam,) + best))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    inf = sub.add_parser("infer")
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--split", default="valid")
    inf.add_argument("--seqs", default=DEV40)
    inf.add_argument("--device", default="cuda")
    sel = sub.add_parser("select")
    sel.add_argument("--dump", default=str(DUMP))
    sel.add_argument("--lam", type=float, default=2.0)
    args = ap.parse_args()
    {"infer": cmd_infer, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
