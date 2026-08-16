"""Lever 1 — occlusion-aware track breaking (post-process). A track is split
wherever its box overlaps another track's box (a crossing/occlusion): the
ambiguous overlap frames are dropped and each occlusion-free run becomes its own
tracklet. Turns impure long tracks into short pure ones. Scored against the
DINOv3/OSNet shipped baselines (IDF1/IDsw) on the fast-test subset."""
import argparse
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import numpy as np
import pandas as pd

from track_bench import GT, replay, score_mot


def occlusion_break(obs, iou_thresh=0.35):
    """Split each track at frames where it overlaps another track; drop overlap
    frames; renumber occlusion-free runs as fresh tracklets."""
    obs = obs.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
    # per-frame: flag boxes that overlap another box
    occ = np.zeros(len(obs), bool)
    for _, g in obs.groupby("frame_idx"):
        idx = g.index.to_numpy()
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        if len(b) < 2:
            continue
        x1 = np.maximum.outer(b[:, 0], b[:, 0]); y1 = np.maximum.outer(b[:, 1], b[:, 1])
        x2 = np.minimum.outer(b[:, 2], b[:, 2]); y2 = np.minimum.outer(b[:, 3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area[:, None] + area[None, :] - inter
        iou = inter / np.maximum(union, 1e-9)
        np.fill_diagonal(iou, 0.0)
        occ[idx] = iou.max(axis=1) >= iou_thresh
    obs = obs[~occ].reset_index(drop=True)  # drop ambiguous overlap frames
    # renumber: within a track, a gap in frames starts a new tracklet
    new_id = np.empty(len(obs), np.int64)
    nxt = 0
    for tid, g in obs.groupby("track_id"):
        fr = g.frame_idx.to_numpy()
        cut = np.concatenate([[True], np.diff(fr) > 1])  # new run after any frame gap
        ids = nxt + np.cumsum(cut) - 1
        new_id[g.index.to_numpy()] = ids
        nxt = ids.max() + 1
    obs["track_id"] = new_id
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT)
    gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]

    params = dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90)
    obs = replay(cache, args.video, params)
    print(f"# lever1 occlusion-break — {args.label}")
    print(f"{'variant':<16}{'IDF1':>7}{'MOTA':>7}{'IDsw':>6}{'Frag':>6}{'#trk':>6}")
    for name, o in [("baseline", obs), ("occ-break", occlusion_break(obs))]:
        s = score_mot(o, gt)
        print(f"{name:<16}{s.idf1:>7.3f}{s.mota:>7.3f}{int(s.num_switches):>6d}{int(s.num_fragmentations):>6d}{o.track_id.nunique():>6d}", flush=True)


if __name__ == "__main__":
    main()
