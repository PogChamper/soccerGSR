"""Error profile of an assembled jersey state: every identity of a GSR_DUMP_MEMBERS dump
classified against GT, with its box mass and vote density.

Fork of vote_density.py with the two defects fixed:
  - an identity whose boxes match no GT box at IoU >= 0.5 was scored as a false commit
    on a GT-None identity; the two are now separate classes (unmatched_commit vs
    false_commit), and the same split is made on the abstaining side;
  - the vote-density floor (GSR_MIN_VOTE_FRAC) is a CLI argument, not a hardcoded ladder.

A dump is the state at the moment the assembler emitted it. --compose-tag additionally
pools its identities through a later tree (dump row i is composed prediction i for
i < len(rows)) and takes the committed jersey from that tree, so one pass profiles both
the pre-link and the composed state.

    python vote_profile.py --dump-dir /mnt/d/jersey-lab/chain_compose/dump_k1jr \
        --cache jnrstar2_u02_c08 --conf 0.5 --floor 0.02 \
        --compose-tag tun_k1jr_sportsl80 --out /mnt/d/jersey-lab/tracklets_v1/profile_6931.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/home/dxdxxd/projects/soccer-app/gsr/out/valid")
PREDS = Path("/home/dxdxxd/projects/soccer-app/gsr/out/preds/SoccerNetGS-valid")
GT = Path("/mnt/d/datasets/soccernet2025/valid")
CLASSES = ("right", "wrong", "false_commit", "unmatched_commit",
           "abstain_known", "abstain_none", "abstain_unmatched")


def iou(p, q):
    x1, y1 = max(p[0], q[0]), max(p[1], q[1]); x2, y2 = min(p[2], q[2]), min(p[3], q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def gt_by_frame(seq):
    """{frame_idx: [(track_id, box, jersey), ...]} over GT player and goalkeeper boxes."""
    g = defaultdict(list)
    for x in json.load(open(GT / seq / "Labels-GameState.json"))["annotations"]:
        if x.get("supercategory") != "object" or x["category_id"] not in (1, 2) or not x.get("bbox_image"):
            continue
        b = x["bbox_image"]
        g[int(x["image_id"][-6:]) - 1].append(
            (x["track_id"], (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]),
             (x.get("attributes") or {}).get("jersey")))
    return g


def classify(pred, matched, gt_jersey):
    if pred is None:
        if not matched:
            return "abstain_unmatched"
        return "abstain_none" if gt_jersey is None else "abstain_known"
    if not matched:
        return "unmatched_commit"
    if gt_jersey is None:
        return "false_commit"
    return "right" if str(pred) == str(gt_jersey) else "wrong"


def profile(scan, key, out_rows, scope):
    """Aggregate the per-row scan into identities of one scope ('base' or 'composed')."""
    by = defaultdict(list)
    for r in scan:
        by[(r["seq"], r[key])].append(r)
    for (seq, oid), rs in sorted(by.items()):
        gts = Counter((r["gt_tid"], r["gt_jersey"]) for r in rs if r["gt_tid"] is not None)
        gt_tid, gt_jersey = gts.most_common(1)[0][0] if gts else (None, None)
        matched = sum(gts.values())
        pred = rs[0][key + "_jersey"]
        votes = sum(r["vote"] for r in rs)
        out_rows.append(dict(
            scope=scope, seq=seq, oid=oid, role=Counter(r["role"] for r in rs).most_common(1)[0][0],
            boxes=len(rs), matched=matched, votes=votes, density=round(votes / len(rs), 4),
            pred=pred, gt_tid=gt_tid, gt_jersey=gt_jersey,
            klass=classify(pred, matched, gt_jersey)))


def table(rows, scope, floor):
    sub = [r for r in rows if r["scope"] == scope]
    boxes = sum(r["boxes"] for r in sub)
    print(f"\nscope={scope}  identities={len(sub)}  boxes={boxes}")
    print(f"  {'class':18s} {'ids':>5s} {'boxes':>8s} {'box share':>10s}")
    agg = []
    for k in CLASSES:
        g = [r for r in sub if r["klass"] == k]
        b = sum(r["boxes"] for r in g)
        agg.append((k, len(g), b, b / boxes))
        print(f"  {k:18s} {len(g):5d} {b:8d} {100 * b / boxes:9.2f}%")
    kill = [r for r in sub if r["pred"] is not None and r["density"] < floor]
    print(f"  floor {floor}: {len(kill)} committed identities below it "
          f"({dict(Counter(r['klass'] for r in kill))}, {sum(r['boxes'] for r in kill)} boxes)")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--cache", required=True, help="jersey cache tag (generic format)")
    ap.add_argument("--conf", type=float, default=0.5, help="GSR_OCR_CONF_TH the state was built with")
    ap.add_argument("--floor", type=float, default=0.0, help="GSR_MIN_VOTE_FRAC to report against")
    ap.add_argument("--compose-tag", default="", help="prediction tag whose tree pools the dump identities")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scan, extra_rows = [], 0
    for mf in sorted(args.dump_dir.glob("SNGS-*.pkl")):
        seq = mf.stem
        base2comp = {}
        rows = pickle.load(open(mf, "rb"))
        jer = pickle.load(open(OUT / seq / f"jersey_{args.cache}.pkl", "rb"))["frames"]
        gt = gt_by_frame(seq)
        comp = []
        if args.compose_tag:
            preds = json.load(open(PREDS / args.compose_tag / "data" / f"{seq}.json"))["predictions"]
            assert len(rows) <= len(preds), f"{seq}: {len(rows)} dump rows vs {len(preds)} predictions"
            extra_rows += len(preds) - len(rows)
            comp = preds[: len(rows)]
            assert all(int(p["image_id"][-6:]) - 1 == r[0] for p, r in zip(comp, rows)), \
                f"{seq}: dump rows are not aligned with {args.compose_tag}"
        for i, (fi, j, base_id, role, _team, jersey, bbox) in enumerate(rows):
            if role not in ("player", "goalkeeper"):
                continue
            nums, confs = jer[fi]
            best = max(((iou(bbox, gb), tid, jn) for tid, gb, jn in gt.get(fi, [])), default=(0, None, None))
            r = dict(seq=seq, base=base_id, base_jersey=jersey, role=role,
                     vote=int(j < len(nums) and int(nums[j]) >= 0 and float(confs[j]) >= args.conf),
                     gt_tid=best[1] if best[0] >= 0.5 else None,
                     gt_jersey=best[2] if best[0] >= 0.5 else None)
            if comp:
                cid = comp[i]["track_id"]
                if base2comp.setdefault(base_id, cid) != cid:
                    raise SystemExit(f"{seq}: base id {base_id} maps to two composed ids")
                r["composed"] = cid
                r["composed_jersey"] = comp[i]["attributes"].get("jersey")
            scan.append(r)

    out_rows = []
    profile(scan, "base", out_rows, "base")
    if args.compose_tag:
        profile(scan, "composed", out_rows, "composed")

    print(f"dump {args.dump_dir}  cache {args.cache}  conf {args.conf}")
    if args.compose_tag:
        print(f"composed through {args.compose_tag}; {extra_rows} interpolated prediction rows "
              f"are outside the dump alignment and not counted in the box mass")
    agg = {s: table(out_rows, s, args.floor) for s in dict.fromkeys(r["scope"] for r in out_rows)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "class", "identities", "boxes", "box_share"])
        for scope, rows_ in agg.items():
            for k, n, b, share in rows_:
                w.writerow([scope, k, n, b, f"{share:.4f}"])
    det = args.out.with_name(args.out.stem + "_identities.csv")
    with det.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {args.out}\nwrote {det}")


if __name__ == "__main__":
    main()
