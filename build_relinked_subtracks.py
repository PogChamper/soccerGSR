"""Full-match Phase-0/1 stack -> LTPI fragments. Take the real tracker's full-match
output, apply the offline stack (occlusion break -> match-adaptive OSNet tracklet
centroids -> k-reciprocal relink under temporal mutual-exclusion), and write the
RELINKED identities as LTPI-format subtracks (majority GT identity per relinked id).
Feeding these to extract_reid_torch + the identification harness gives the end-to-end
CSIS the tracking rebuild actually buys, vs the raw-tracker 0.43 baseline."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

import cv2
import numpy as np
import pandas as pd

from extract_reid_torch import build_osnet
from ltpi_research.advanced import k_reciprocal_rerank

GT_ROOT = Path("/home/dxdxxd/projects/football/data-ltpi/LTPI dataset/ds_ltpi/test")


def occlusion_break(obs, iou_thresh=0.35):
    """Split tracks at box-overlap (crossing) frames into occlusion-free tracklets."""
    obs = obs.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)
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
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        np.fill_diagonal(iou, 0.0)
        occ[idx] = iou.max(axis=1) >= iou_thresh
    obs = obs[~occ].reset_index(drop=True)
    new_id = np.empty(len(obs), np.int64); nxt = 0
    for tid, g in obs.groupby("track_id"):
        fr = g.frame_idx.to_numpy()
        cut = np.concatenate([[True], np.diff(fr) > 1])
        ids = nxt + np.cumsum(cut) - 1
        new_id[g.index.to_numpy()] = ids
        nxt = ids.max() + 1
    obs = obs.copy(); obs["track_id"] = new_id
    return obs


def iou_to(gb, box):
    ix1, iy1 = np.maximum(gb[:, 0], box[0]), np.maximum(gb[:, 1], box[1])
    ix2, iy2 = np.minimum(gb[:, 2], box[2]), np.minimum(gb[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]); ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--half", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-vid", default="relink2")
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--max-crops", type=int, default=24)
    ap.add_argument("--th", type=float, default=0.20)
    args = ap.parse_args()

    obs = pd.read_csv(args.obs)
    obs = obs[obs.cls_id == 0].copy().reset_index(drop=True)
    obs["orig"] = np.arange(len(obs))
    broken = occlusion_break(obs).reset_index(drop=True)
    keep = broken.groupby("track_id").frame_idx.transform("size") >= args.min_frames
    broken = broken[keep].reset_index(drop=True)
    print(f"broken tracklets: {broken.track_id.nunique()}", flush=True)

    # sample crops per tracklet, decode video once, embed with match-adaptive OSNet
    emb = build_osnet(args.model_path, device="cuda")
    sampled = []
    for tid, g in broken.groupby("track_id"):
        g = g.sort_values("frame_idx")
        idx = np.unique(np.linspace(0, len(g) - 1, min(args.max_crops, len(g))).round().astype(int))
        sampled.append(g.iloc[idx].assign(_tid=tid))
    S = pd.concat(sampled)
    by_frame = {f: gg for f, gg in S.groupby("frame_idx")}
    cap = cv2.VideoCapture(args.video)
    feats = {}  # tid -> list of embeddings
    fmax = int(S.frame_idx.max())
    for fi in range(fmax + 1):
        if fi not in by_frame:
            cap.grab(); continue
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        crops, tids = [], []
        for _, o in by_frame[fi].iterrows():
            x1, y1, x2, y2 = max(0, int(o.x1)), max(0, int(o.y1)), min(w, int(o.x2)), min(h, int(o.y2))
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crops.append(frame[y1:y2, x1:x2]); tids.append(int(o._tid))
        if crops:
            e = emb.embed(crops)
            for t, v in zip(tids, e):
                feats.setdefault(t, []).append(v)
    cap.release()

    tl = [(t, np.mean(v, 0) / max(np.linalg.norm(np.mean(v, 0)), 1e-9),
           int(broken[broken.track_id == t].frame_idx.min()), int(broken[broken.track_id == t].frame_idx.max()))
          for t, v in feats.items()]
    tids = [t[0] for t in tl]
    cents = np.stack([t[1] for t in tl]).astype(np.float32)
    spans = [(t[2], t[3]) for t in tl]
    order = sorted(range(len(tl)), key=lambda i: spans[i][0] - spans[i][1])  # longest first
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    print(f"relinking {len(tl)} tracklets ...", flush=True)

    clusters, assign = [], {}
    for i in order:
        f0, f1 = spans[i]; best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if any(not (f1 < s0 or f0 > s1) for s0, s1 in cl["spans"]):
                continue
            s = max(rr[i, m] for m in cl["members"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= args.th:
            clusters[best]["members"].append(i); clusters[best]["spans"].append((f0, f1)); assign[tids[i]] = best
        else:
            clusters.append(dict(members=[i], spans=[(f0, f1)])); assign[tids[i]] = len(clusters) - 1
    print(f"relinked -> {len(clusters)} identities", flush=True)
    broken["relink_id"] = broken.track_id.map(assign)

    # IoU-match each box to GT -> majority GT identity per relinked id
    gt = pd.read_csv(GT_ROOT / args.half / "subtracks.csv")
    gt["identity"] = gt.team_global * 100 + gt.jersey_number
    broken["image_id"] = broken.frame_idx + 1
    gt_by = {f: g for f, g in gt.groupby("image_id")}
    match = np.full(len(broken), -1, np.int64)
    br = broken.reset_index(drop=True)
    for img, g in br.groupby("image_id"):
        gtf = gt_by.get(img)
        if gtf is None:
            continue
        gb = gtf[["x1", "y1", "x2", "y2"]].to_numpy(float); gid = gtf["identity"].to_numpy()
        for i, o in g.iterrows():
            iou = iou_to(gb, np.array([o.x1, o.y1, o.x2, o.y2], float)); j = int(iou.argmax())
            if iou[j] >= 0.5:
                match[i] = gid[j]
    br["match_id"] = match

    rows = []
    for rid, g in br.groupby("relink_id"):
        m = g[g.match_id >= 0]
        if len(m) == 0:
            continue
        maj = int(m.match_id.value_counts().index[0]); tg = maj // 100; jn = maj % 100
        team = "left" if tg == 1 else "right"
        for _, o in g.iterrows():
            rows.append(dict(image_id=int(o.image_id), x1=o.x1, y1=o.y1, x2=o.x2, y2=o.y2,
                             team=team, jersey_number=jn, gt_track_id=maj, subtrack_id=int(rid), team_global=tg))
    sub = pd.DataFrame(rows).sort_values(["image_id", "subtrack_id"])
    out = Path(args.out_root) / args.out_vid
    out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "subtracks.csv", index=False)
    for f in ("roster.csv", "substitutions.csv"):
        shutil.copy(GT_ROOT / args.half / f, out / f)
    vl = out / "video.mp4"
    if not vl.exists():
        vl.symlink_to(GT_ROOT / args.half / "video.mp4")
    print(f"[relink] {sub.subtrack_id.nunique()} relinked fragments, {sub['gt_track_id'].nunique()} identities -> {out}")


if __name__ == "__main__":
    main()
