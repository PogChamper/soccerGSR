"""P1: tracklet dataset v1 - full-rate logits per GT track, stride-8 GT crops, one manifest.

Three subcommands, all CPU and resumable per clip:

    python gsr/jersey_track/build_tracklets.py logits --split train
    python gsr/jersey_track/build_tracklets.py logits --split valid --seqs valid12
    python gsr/jersey_track/build_tracklets.py crops --seqs tail17
    python gsr/jersey_track/build_tracklets.py manifest

`logits` matches every player/gk detection to a GT track (IoU >= 0.5, mine_train_crops.match_seq)
and stores the raw gate/OCR logits, geometry and a per-track index in one npz per clip; the
unmatched detections go into a second array group (the abstain-side distribution at inference).
`crops` re-cuts the stride-8 GT player crops the ltpi cache is missing. `manifest` joins the
per-track index of the 57 train clips with the crop counts of both sources.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr/jersey_track")

import cv2
import numpy as np
import pandas as pd

import common as C
from mine_train_crops import match_seq

ROOT = Path("/mnt/d/jersey-lab/tracklets_v1")
POOL_CSV = Path("/mnt/d/jersey-lab/pool/gt_crops_scores_train.csv")
VALID12 = [f"SNGS-{i:03d}" for i in range(21, 33)]
TAIL17 = [f"SNGS-{i:03d}" for i in range(154, 171)]
ROLE_CODE = {"player": 0, "goalkeeper": 1}
ROLE_NAME = {v: k for k, v in ROLE_CODE.items()}


def gt_tracks(labels):
    """track_id -> (role_code, label, team, n_boxes) for every player/gk GT track.

    label is the jersey number or -1 for GT-None. Attributes are constant per track.
    """
    out = {}
    for a in labels["annotations"]:
        attr = a.get("attributes")
        tid = a.get("track_id")
        if not attr or tid is None or attr.get("role") not in ROLE_CODE:
            continue
        tid = int(tid)
        if tid not in out:
            jersey = attr.get("jersey")
            out[tid] = [ROLE_CODE[attr["role"]], int(jersey) if jersey is not None else -1,
                        attr.get("team") or "", 0]
        out[tid][3] += 1
    return out


def logits_seq(split, seq, out, iou_th):
    dst = out / f"{seq}.npz"
    if dst.exists():
        return None
    t0 = time.time()
    df, _ = match_seq(split, seq, iou_th)
    jer = C.load(C.OUT_ROOT / split / seq / "jersey.pkl")["frames"]
    image_ids = C.load(C.OUT_ROOT / split / seq / "det.pkl")["image_ids"]
    tracks = gt_tracks(C.load_labels(split, seq))

    fi = df["frame_idx"].to_numpy(np.int32)
    di = df["det_idx"].to_numpy(np.int32)
    vis = np.array([jer[f][0][d] for f, d in zip(fi, di)], np.float32)
    tens = np.stack([jer[f][1][d] for f, d in zip(fi, di)]).astype(np.float32)
    units = np.stack([jer[f][2][d] for f, d in zip(fi, di)]).astype(np.float32)
    box = df[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
    conf = df["det_conf"].to_numpy(np.float32)
    scored = df["scored"].to_numpy(bool)
    matched = df["matched"].to_numpy(bool)
    # detector class 0/1 also names the role of an unmatched box; matched rows take the GT role
    role = df["det_cls"].to_numpy(np.uint8)
    trk = np.zeros(len(df), np.int32)
    iou = np.zeros(len(df), np.float32)
    if matched.any():
        role[matched] = [ROLE_CODE[r] for r in df.loc[matched, "gt_role"]]
        trk[matched] = df.loc[matched, "gt_track"].to_numpy(np.int32)
        iou[matched] = df.loc[matched, "iou"].to_numpy(np.float32)

    m = np.where(matched)[0]
    m = m[np.lexsort((fi[m], trk[m]))]  # contiguous per-track slices, time-ordered
    u = np.where(~matched)[0]

    ids = sorted(tracks)
    starts = np.searchsorted(trk[m], ids, "left").astype(np.int32)
    ends = np.searchsorted(trk[m], ids, "right").astype(np.int32)

    np.savez_compressed(
        dst,
        seq=seq, split=split, iou_th=np.float32(iou_th), image_ids=np.array(image_ids),
        m_frame=fi[m], m_det=di[m].astype(np.int16), m_track=trk[m], m_role=role[m],
        m_vis=vis[m], m_tens=tens[m], m_units=units[m], m_box=box[m], m_conf=conf[m],
        m_iou=iou[m], m_scored=scored[m],
        u_frame=fi[u], u_det=di[u].astype(np.int16), u_role=role[u], u_vis=vis[u],
        u_tens=tens[u], u_units=units[u], u_box=box[u], u_conf=conf[u], u_scored=scored[u],
        t_track=np.array(ids, np.int32),
        t_role=np.array([tracks[i][0] for i in ids], np.uint8),
        t_label=np.array([tracks[i][1] for i in ids], np.int16),
        t_n_gt=np.array([tracks[i][3] for i in ids], np.int32),
        t_start=starts, t_end=ends)
    known = sum(1 for i in ids if tracks[i][0] == 0 and tracks[i][1] >= 0)
    print(f"[logits {seq}] dets {len(df)} matched {len(m)} unmatched {len(u)}; "
          f"tracks {len(ids)} ({known} known player) in {time.time() - t0:.1f} s", flush=True)
    return len(df), len(m), len(u), len(ids)


def crops_seq(seq, out, stride):
    idx_dir = out / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    dst = idx_dir / f"{seq}.csv"
    if dst.exists():
        return None
    t0 = time.time()
    labels = C.load_labels("train", seq)
    tracks = gt_tracks(labels)
    per_image = {}
    for a in labels["annotations"]:
        if a.get("category_id") != 1 or a.get("track_id") is None:
            continue
        per_image.setdefault(a["image_id"], []).append((int(a["track_id"]), a["bbox_image"]))
    images = sorted(labels["images"], key=lambda im: im["image_id"])[::stride]

    sd = C.seq_dir("train", seq)
    rows, tiny = [], 0
    for im in images:
        anns = per_image.get(im["image_id"], [])
        if not anns:
            continue
        frame = cv2.imread(str(sd / "img1" / im["file_name"]))
        if frame is None:
            print(f"[crops {seq}] missing frame {im['file_name']}", flush=True)
            continue
        fh, fw = frame.shape[:2]
        for tid, b in anns:
            x1, y1 = int(max(0, b["x"])), int(max(0, b["y"]))
            x2, y2 = int(min(fw, b["x"] + b["w"])), int(min(fh, b["y"] + b["h"]))
            if x2 - x1 < 4 or y2 - y1 < 8:
                tiny += 1
                continue
            d = out / f"{seq}_{tid}"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{seq}_{im['image_id']}.jpg"
            cv2.imwrite(str(path), frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            role, label, team, _ = tracks[tid]
            rows.append({"seq": seq, "track_id": tid, "image_id": im["image_id"], "path": str(path),
                         "gt_jersey": "" if label < 0 else label, "role": ROLE_NAME[role],
                         "team": team, "w": x2 - x1, "h": y2 - y1})
    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    print(f"[crops {seq}] {len(df)} crops, {df['track_id'].nunique()} tracks, {len(images)} frames, "
          f"{tiny} too small, in {time.time() - t0:.1f} s", flush=True)
    return df


def cmd_logits(args):
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.seqs == "all":
        seqs = C.list_seqs(args.split)
    elif args.seqs == "valid12":
        seqs = VALID12
    else:
        seqs = args.seqs.split(",")
    seqs = [s for s in seqs if (C.OUT_ROOT / args.split / s / "jersey.pkl").exists()]
    print(f"[logits] {len(seqs)} {args.split} clips -> {out}", flush=True)
    for seq in seqs:
        logits_seq(args.split, seq, out, args.iou)
    summarise_logits(out)


def summarise_logits(out):
    n_det = n_m = n_u = n_trk = n_known = n_none = 0
    files = sorted(out.glob("SNGS-*.npz"))
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            n_m += len(z["m_frame"])
            n_u += len(z["u_frame"])
            n_trk += len(z["t_track"])
            pl = z["t_role"] == 0
            n_known += int((pl & (z["t_label"] >= 0)).sum())
            n_none += int((pl & (z["t_label"] < 0)).sum())
    n_det = n_m + n_u
    gb = sum(f.stat().st_size for f in files) / 2**30
    print(f"[logits] {len(files)} files, {n_det} detections ({n_m} matched / {n_u} unmatched), "
          f"{n_trk} GT tracks ({n_known} known player / {n_none} None player), {gb:.2f} GB", flush=True)


def cmd_crops(args):
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    seqs = TAIL17 if args.seqs == "tail17" else args.seqs.split(",")
    print(f"[crops] {len(seqs)} clips, stride {args.stride} -> {out}", flush=True)
    for seq in seqs:
        crops_seq(seq, out, args.stride)
    idx = pd.concat([pd.read_csv(p) for p in sorted((out / "_index").glob("SNGS-*.csv"))],
                    ignore_index=True)
    idx.to_csv(out / "index.csv", index=False)
    known = idx[idx["gt_jersey"].notna()].groupby(["seq", "track_id"]).ngroups
    print(f"[crops] index {out / 'index.csv'}: {len(idx)} crops, "
          f"{idx.groupby(['seq', 'track_id']).ngroups} tracks ({known} with a number), "
          f"h median {idx['h'].median():.0f} px", flush=True)


def cmd_manifest(args):
    rows = []
    for f in sorted((ROOT / "logits_train").glob("SNGS-*.npz")):
        with np.load(f, allow_pickle=False) as z:
            seq = str(z["seq"])
            for tid, role, label, n_gt, s, e in zip(z["t_track"], z["t_role"], z["t_label"],
                                                    z["t_n_gt"], z["t_start"], z["t_end"]):
                rows.append({"seq": seq, "gt_track": int(tid), "label": "" if label < 0 else int(label),
                             "role": ROLE_NAME[int(role)], "n_boxes_gt": int(n_gt),
                             "n_dets_matched": int(e - s)})
    man = pd.DataFrame(rows)

    pool = pd.read_csv(POOL_CSV, usecols=["seq_name", "track_id"])
    pool_n = pool.groupby(["seq_name", "track_id"]).size()
    tail = pd.read_csv(ROOT / "crops_tail17" / "index.csv", usecols=["seq", "track_id"])
    tail_n = tail.groupby(["seq", "track_id"]).size()
    pool_seqs, tail_seqs = set(pool["seq_name"]), set(tail["seq"])

    key = list(zip(man["seq"], man["gt_track"]))
    man["crop_source"] = ["pool" if s in pool_seqs else "tail17" if s in tail_seqs else ""
                          for s in man["seq"]]
    man["n_crops_stride8"] = [int(pool_n.get(k, 0)) + int(tail_n.get(k, 0)) for k in key]
    man = man[["seq", "gt_track", "label", "role", "n_boxes_gt", "n_dets_matched",
               "n_crops_stride8", "crop_source"]]
    dst = ROOT / "manifest_tracks.csv"
    man.to_csv(dst, index=False)

    pl = man[man["role"] == "player"]
    known, none = int((pl["label"] != "").sum()), int((pl["label"] == "").sum())
    gk = man[man["role"] == "goalkeeper"]
    print(f"[manifest] {dst}: {len(man)} tracks over {man['seq'].nunique()} clips", flush=True)
    print(f"[manifest] player known {known} (target 933), player None {none} (target 212), "
          f"goalkeeper {len(gk)} ({int((gk['label'] != '').sum())} with a number)", flush=True)
    print(f"[manifest] GT boxes {int(man['n_boxes_gt'].sum())} (known player "
          f"{int(pl.loc[pl['label'] != '', 'n_boxes_gt'].sum())}), matched dets "
          f"{int(man['n_dets_matched'].sum())}, crops {int(man['n_crops_stride8'].sum())} "
          f"(pool {int(man.loc[man['crop_source'] == 'pool', 'n_crops_stride8'].sum())} / "
          f"tail17 {int(man.loc[man['crop_source'] == 'tail17', 'n_crops_stride8'].sum())}), "
          f"tracks without crops {int((man['n_crops_stride8'] == 0).sum())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("logits", help="per-clip full-rate logits npz")
    p.add_argument("--split", default="train")
    p.add_argument("--seqs", default="all", help="'all', 'valid12' or a comma list")
    p.add_argument("--out", default="logits_train", help="subdirectory of the dataset root")
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(fn=cmd_logits)

    p = sub.add_parser("crops", help="stride-8 GT player crops")
    p.add_argument("--seqs", default="tail17", help="'tail17' or a comma list")
    p.add_argument("--out", default="crops_tail17")
    p.add_argument("--stride", type=int, default=8)
    p.set_defaults(fn=cmd_crops)

    p = sub.add_parser("manifest", help="one row per train GT track")
    p.set_defaults(fn=cmd_manifest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
