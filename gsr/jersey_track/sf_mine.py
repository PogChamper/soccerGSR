"""Mine confirmed jersey crops from SoccerFactory parts 1-2 (kit diversity stream).

Numbered boxes above --legibility from the per-clip annotation JSON, capped per
clip and spread over frames and detector ids; sequential decode per clip (one
pass, grab/retrieve). Output: jpgs + a manifest csv compatible with the external
single-crop stream of tracklet_pix / the 23-40-49 recipe. Resumable per clip via
per-clip manifests. CPU only; run several shards in parallel.

    python sf_mine.py --shard 0 --shards 5 --legibility 0.6 --cap 40
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

ANN = Path("/mnt/d/datasets/football/annotations")
VID = [Path("/mnt/d/datasets/football/soccer_factory_videos_part1/videos"),
       Path("/mnt/d/datasets/football/soccer_factory_videos_part2/videos")]
OUT = Path("/mnt/d/jersey-lab/sf_mine_v1")


def mine_clip(clip: str, vp: Path, leg: float, cap_n: int, rng: random.Random) -> list[dict]:
    d = json.load(open(ANN / f"{clip}.json"))
    cands = []
    for fk, fr in d.items():
        for p in (fr.get("people") or []):
            n = p.get("jersey_number")
            if n is not None and (p.get("legibility_score") or 0) >= leg:
                x, y, w, h = p["bbox_ltwh"]
                if w >= 12 and h >= 24:
                    cands.append((int(fk), int(p["id"]), p["bbox_ltwh"], str(n)))
    if not cands:
        return []
    rng.shuffle(cands)
    picked, seen = [], set()
    for fk, pid, bb, n in sorted(cands, key=lambda c: (c[0] in {p[0] for p in picked}, 0)):
        if len(picked) >= cap_n:
            break
        key = (fk // 25, n)  # at most one crop of a number per second
        if key in seen:
            continue
        seen.add(key)
        picked.append((fk, pid, bb, n))
    picked.sort()
    cap = cv2.VideoCapture(str(vp))
    rows, want, wi = [], [p[0] for p in picked], 0
    fi = 0
    while wi < len(picked):
        ok = cap.grab()
        if not ok:
            break
        fi += 1
        while wi < len(picked) and picked[wi][0] == fi:
            ok, frame = cap.retrieve()
            if not ok:
                wi += 1
                continue
            fk, pid, (x, y, w, h), n = picked[wi]
            H, W = frame.shape[:2]
            x1, y1 = int(max(0, x)), int(max(0, y))
            x2, y2 = int(min(W, x + w)), int(min(H, y + h))
            if x2 - x1 >= 12 and y2 - y1 >= 24:
                name = f"{clip}_{fk}_{pid}.jpg"
                cv2.imwrite(str(OUT / "crops" / name), frame[y1:y2, x1:x2],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(name=name, clip=clip, frame=fk, label=n, h=y2 - y1))
            wi += 1
    cap.release()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--legibility", type=float, default=0.6)
    ap.add_argument("--cap", type=int, default=40)
    args = ap.parse_args()
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)
    clips = sorted({p.stem: p for v in VID for p in v.glob("*.mp4")}.items())
    for i, (clip, vp) in enumerate(clips):
        if i % args.shards != args.shard:
            continue
        mf = OUT / "manifests" / f"{clip}.csv"
        if mf.exists():
            continue
        rows = mine_clip(clip, vp, args.legibility, args.cap, random.Random(hash(clip) & 0xFFFF))
        with open(mf, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "clip", "frame", "label", "h"])
            w.writeheader()
            w.writerows(rows)
        print(f"[{clip}] {len(rows)} crops", flush=True)


if __name__ == "__main__":
    main()
