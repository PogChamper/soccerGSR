"""B2 step 1 — build a self-supervised match-adaptive training set from the real
tracker output (no ground truth). Break the full-match tracks into occlusion-free
(pure) tracklets; each becomes a pseudo-identity, its majority team label taken
from the app team classifier. Same-team tracklets are distinct pseudo-ids, so the
intra-team PK sampler + batch-hard triplet pulls identical-kit teammates apart —
exactly the discrimination the online tracker lacks. Leak-free: labels come only
from the tracker's own temporal continuity, never from GT."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import cv2
import numpy as np
import pandas as pd

from lever1_break import occlusion_break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="app observations.csv (full match)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="crop output dir")
    ap.add_argument("--min-frames", type=int, default=16)
    ap.add_argument("--max-crops", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(args.obs)
    obs = obs[obs.cls_id == 0].copy()  # players
    obs["orig"] = np.arange(len(obs))
    team_of_orig = obs.set_index("orig").team_id.to_dict()
    broken = occlusion_break(obs).reset_index(drop=True)  # pure tracklets, renumbered track_id, keeps 'orig'
    broken["team_id"] = broken.orig.map(team_of_orig)

    # keep tracklets with enough frames; sample up to max_crops uniformly
    keep = broken.groupby("track_id").frame_idx.transform("size") >= args.min_frames
    broken = broken[keep]
    sampled = []
    for tid, g in broken.groupby("track_id"):
        g = g.sort_values("frame_idx")
        idx = np.linspace(0, len(g) - 1, min(args.max_crops, len(g))).round().astype(int)
        sampled.append(g.iloc[np.unique(idx)])
    S = pd.concat(sampled).reset_index(drop=True)
    team_maj = S.groupby("track_id").team_id.agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else 0)

    # decode video once, crop the sampled (frame, box)
    by_frame = {f: g for f, g in S.groupby("frame_idx")}
    cap = cv2.VideoCapture(args.video)
    fmax = int(S.frame_idx.max())
    rows = []
    for fi in range(fmax + 1):
        if fi not in by_frame:
            cap.grab(); continue
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        for _, o in by_frame[fi].iterrows():
            x1, y1 = max(0, int(o.x1)), max(0, int(o.y1))
            x2, y2 = min(w, int(o.x2)), min(h, int(o.y2))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            tid = int(o.track_id)
            d = out / str(tid); d.mkdir(exist_ok=True)
            p = d / f"{fi}.jpg"
            cv2.imwrite(str(p), frame[y1:y2, x1:x2])
            rows.append(dict(path=str(p), identity=tid, team=int(team_maj.get(tid, 0))))
    cap.release()
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)
    print(f"[b2] {len(man)} crops, {man.identity.nunique()} pseudo-identities, "
          f"{man.team.nunique()} teams -> {out}/manifest.csv")


if __name__ == "__main__":
    main()
