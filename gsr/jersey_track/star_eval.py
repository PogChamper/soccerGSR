"""Evaluate an 11-class (0-9 + '*') two-head reader without the visibility gate.

Part A: the 500 human-labelled crops (CVAT task 69): abstain rate on human 'no', read accuracy on 'yes'/'partial'.
Part B: valid-12 generic caches for the k1 chain: number where both heads are non-star (conf = min softmax) else -1;
        two variants - no gate (tag <name>_nogate) and AND the deployed gate at 0.7 (tag <name>_gate).

Env soccer_eda:  python gsr/jersey_track/star_eval.py --onnx /mnt/d/jersey-lab/runs/star_v1/reader.onnx --name star_v1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")

import cv2
import numpy as np
import pandas as pd

from app.utils import cuda_env

cuda_env.bootstrap()
import onnxruntime as ort

import common as C

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
STAR = 10


def prep(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    a = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def num_match(h, p):
    if h is None or (isinstance(h, float) and np.isnan(h)):
        return np.nan
    h = str(h).strip()
    if h in ("", "*"):
        return np.nan
    if "*" not in h:
        try:
            return int(h) == int(p)
        except Exception:
            return np.nan
    p = str(int(p))
    return len(h) == len(p) and all(a == "*" or a == b for a, b in zip(h, p))


def decode(pt, pu):
    t, u = int(pt.argmax()), int(pu.argmax())
    if t == STAR or u == STAR:
        return -1, float(min(pt.max(), pu.max())), (t, u)
    return (u if t == 0 else t * 10 + u), float(min(pt.max(), pu.max())), (t, u)


def part_a(so, name):
    d = pd.read_csv("/mnt/d/jersey-lab/cvat_legibility_v1/manifest_with_human.csv")
    root = Path("/mnt/d/jersey-lab/cvat_legibility_v1/images")
    preds, confs, stars = [], [], []
    for r in d.itertuples():
        img = cv2.imread(str(root / r.name))
        # the CVAT images are the crops upscaled x3; the reader input is a 224 stretch, so this is equivalent
        lt, lu = so.run(None, {"input": prep(img)[None]})
        n, c, (t, u) = decode(softmax(lt)[0], softmax(lu)[0])
        preds.append(n); confs.append(c); stars.append("".join("*" if x == STAR else str(x) for x in (t, u)))
    d["star_pred"], d["star_conf"], d["star_heads"] = preds, confs, stars
    leg = d["human_visible"].isin(["yes", "partial"])
    no = d["human_visible"] == "no"
    lines = [f"# {name}: 11-class reader on the 500 human-labelled crops", "",
             f"- human 'no' (n {int(no.sum())}): reader abstains (any head *) on {(d.loc[no, 'star_pred'] < 0).mean():.2f}; reads a number on {(d.loc[no, 'star_pred'] >= 0).mean():.2f}, of which conf >= 0.95: {int(((d.loc[no, 'star_pred'] >= 0) & (d.loc[no, 'star_conf'] >= 0.95)).sum())}",
             f"- deployed ConvNeXt on the same 'no' crops: conf >= 0.95 on {int((d.loc[no, 'cnx_conf'] >= 0.95).sum())} (gate-pass and conf >= 0.95: {int(((d.loc[no, 'vis_p'] >= 0.7) & (d.loc[no, 'cnx_conf'] >= 0.95)).sum())})"]
    dl = d[leg].copy()
    dl["ok"] = [num_match(h, p) if p >= 0 else False for h, p in zip(dl["human_number"], dl["star_pred"])]
    dl["ok_cnx"] = [num_match(h, p) for h, p in zip(dl["human_number"], dl["cnx_pred"])]
    dl = dl[dl["ok_cnx"].notna()]
    lines += [f"- human legible (yes+partial, n {len(dl)}): reader abstains on {(dl['star_pred'] < 0).mean():.2f}; correct number {dl['ok'].mean():.2f} of all legible; among its reads: {dl.loc[dl['star_pred'] >= 0, 'ok'].mean():.2f} (n {int((dl['star_pred'] >= 0).sum())}); at conf >= 0.95: {dl.loc[(dl['star_pred'] >= 0) & (dl['star_conf'] >= 0.95), 'ok'].mean():.2f} (n {int(((dl['star_pred'] >= 0) & (dl['star_conf'] >= 0.95)).sum())})",
              f"- deployed ConvNeXt on the same legible crops: argmax {dl['ok_cnx'].mean():.2f}; gate-pass and conf >= 0.95: {dl.loc[(dl['vis_p'] >= 0.7) & (dl['cnx_conf'] >= 0.95), 'ok_cnx'].mean():.2f} (n {int(((dl['vis_p'] >= 0.7) & (dl['cnx_conf'] >= 0.95)).sum())})",
              f"- human 'yes' only (n {int((dl['human_visible'] == 'yes').sum())}): reader correct {dl.loc[dl['human_visible'] == 'yes', 'ok'].mean():.2f}, abstains {(dl.loc[dl['human_visible'] == 'yes', 'star_pred'] < 0).mean():.2f}"]
    d.to_csv(f"/mnt/d/jersey-lab/cvat_legibility_v1/human_vs_{name}.csv", index=False)
    return lines


def part_b(so, name, split="valid", seqs=None):
    seqs = seqs or [f"SNGS-{i:03d}" for i in range(21, 33)]
    n_read = n_gate = 0
    for seq in seqs:
        det = C.load(C.OUT_ROOT / split / seq / "det.pkl")
        jer = C.load(C.OUT_ROOT / split / seq / "jersey.pkl")["frames"]
        labels = C.load_labels(split, seq)
        _, files = C.frame_index(labels)
        sd = C.seq_dir(split, seq)
        f_nogate, f_gate = [], []
        for fi, (boxes, (vis, _, _)) in enumerate(zip(det["frames"], jer)):
            n = len(boxes)
            nums, confs = np.full(n, -1, np.int32), np.zeros(n, np.float32)
            nums2, confs2 = nums.copy(), confs.copy()
            if n:
                frame = cv2.imread(str(sd / "img1" / files[fi]))
                if frame is not None:
                    H, W = frame.shape[:2]
                    crops, idx = [], []
                    for j, b in enumerate(boxes):
                        if vis[j] <= -1e8:
                            continue
                        x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3]))
                        if x2 - x1 < 4 or y2 - y1 < 8:
                            continue
                        crops.append(prep(frame[y1:y2, x1:x2])); idx.append(j)
                    if idx:
                        lt, lu = so.run(None, {"input": np.stack(crops).astype(np.float32)})
                        pt, pu = softmax(lt), softmax(lu)
                        for k, j in enumerate(idx):
                            num, c, _ = decode(pt[k], pu[k])
                            if num >= 0:
                                nums[j], confs[j] = num, c
                                n_read += 1
                                if 1 / (1 + np.exp(-vis[j])) >= 0.7:
                                    nums2[j], confs2[j] = num, c
                                    n_gate += 1
            f_nogate.append((nums, confs)); f_gate.append((nums2, confs2))
        C.dump({"frames": f_nogate}, C.OUT_ROOT / split / seq / f"jersey_{name}_nogate.pkl")
        C.dump({"frames": f_gate}, C.OUT_ROOT / split / seq / f"jersey_{name}_gate.pkl")
        print(f"[star {seq}] cached", flush=True)
    return [f"- valid-12 caches: {n_read} non-star reads without gate, {n_gate} with the deployed gate at 0.7 -> jersey_{name}_nogate.pkl / jersey_{name}_gate.pkl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--seqs", default="", help="comma list for part B (default valid-12)")
    args = ap.parse_args()
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    so = ort.InferenceSession(args.onnx, providers=prov)
    lines = part_a(so, args.name)
    if not args.skip_b:
        lines += part_b(so, args.name, seqs=args.seqs.split(",") if args.seqs else None)
    out = Path(f"/mnt/d/jersey-lab/runs/{args.name}/star_eval.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
