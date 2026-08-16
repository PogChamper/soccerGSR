"""Metric bridge: write the tracker's OCCLUSION-BROKEN tracklets (short, pure,
many-per-identity — the analog of GT subtracks) as LTPI-format fragments, so our
identification (gallery + offline stack) can name them and we get a CSIS that is
apples-to-apples with the GT-fragment ceiling. No relink here (the identification
IS the linking) and no long tracks (which broke the enrollment protocol)."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
import numpy as np
import pandas as pd

GT_ROOT = Path("/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test")


def occlusion_break(obs, iou_thresh=0.35):
    obs = obs.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    occ = np.zeros(len(obs), bool)
    for _, g in obs.groupby("frame_idx"):
        idx = g.index.to_numpy(); b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        if len(b) < 2:
            continue
        x1 = np.maximum.outer(b[:, 0], b[:, 0]); y1 = np.maximum.outer(b[:, 1], b[:, 1])
        x2 = np.minimum.outer(b[:, 2], b[:, 2]); y2 = np.minimum.outer(b[:, 3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        np.fill_diagonal(iou, 0.0)
        occ[idx] = iou.max(axis=1) >= iou_thresh
    obs = obs[~occ].reset_index(drop=True)
    new = np.empty(len(obs), np.int64); nxt = 0
    for tid, g in obs.groupby("track_id"):
        fr = g.frame_idx.to_numpy(); cut = np.concatenate([[True], np.diff(fr) > 1])
        ids = nxt + np.cumsum(cut) - 1; new[g.index.to_numpy()] = ids; nxt = ids.max() + 1
    obs = obs.copy(); obs["track_id"] = new
    return obs


def iou_to(gb, box):
    ix1, iy1 = np.maximum(gb[:, 0], box[0]), np.maximum(gb[:, 1], box[1])
    ix2, iy2 = np.minimum(gb[:, 2], box[2]), np.minimum(gb[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]); ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True); ap.add_argument("--half", required=True)
    ap.add_argument("--out-root", required=True); ap.add_argument("--out-vid", default="broken2")
    ap.add_argument("--min-frames", type=int, default=10)
    args = ap.parse_args()

    obs = pd.read_csv(args.obs); obs = obs[obs.cls_id == 0].copy().reset_index(drop=True)
    broken = occlusion_break(obs).reset_index(drop=True)
    keep = broken.groupby("track_id").frame_idx.transform("size") >= args.min_frames
    broken = broken[keep].reset_index(drop=True)
    print(f"broken tracklets: {broken.track_id.nunique()}", flush=True)

    gt = pd.read_csv(GT_ROOT / args.half / "subtracks.csv"); gt["identity"] = gt.team_global * 100 + gt.jersey_number
    broken["image_id"] = broken.frame_idx + 1
    gt_by = {f: g for f, g in gt.groupby("image_id")}
    match = np.full(len(broken), -1, np.int64)
    for img, g in broken.groupby("image_id"):
        gtf = gt_by.get(img)
        if gtf is None:
            continue
        gb = gtf[["x1", "y1", "x2", "y2"]].to_numpy(float); gid = gtf["identity"].to_numpy()
        for i, o in g.iterrows():
            iou = iou_to(gb, np.array([o.x1, o.y1, o.x2, o.y2], float)); j = int(iou.argmax())
            if iou[j] >= 0.5:
                match[i] = gid[j]
    broken["match_id"] = match

    rows = []
    for tid, g in broken.groupby("track_id"):
        m = g[g.match_id >= 0]
        if len(m) == 0:
            continue
        maj = int(m.match_id.value_counts().index[0]); tg = maj // 100; jn = maj % 100
        team = "left" if tg == 1 else "right"
        for _, o in g.iterrows():
            rows.append(dict(image_id=int(o.image_id), x1=o.x1, y1=o.y1, x2=o.x2, y2=o.y2,
                             team=team, jersey_number=jn, gt_track_id=maj, subtrack_id=int(tid), team_global=tg))
    sub = pd.DataFrame(rows).sort_values(["image_id", "subtrack_id"])
    out = Path(args.out_root) / args.out_vid; out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "subtracks.csv", index=False)
    for f in ("roster.csv", "substitutions.csv"):
        shutil.copy(GT_ROOT / args.half / f, out / f)
    vl = out / "video.mp4"
    if not vl.exists():
        vl.symlink_to(GT_ROOT / args.half / "video.mp4")
    fps = sub.groupby("gt_track_id").subtrack_id.nunique()
    print(f"[broken] {sub.subtrack_id.nunique()} fragments, {sub['gt_track_id'].nunique()} identities, "
          f"~{fps.mean():.1f} fragments/identity -> {out}")


if __name__ == "__main__":
    main()
