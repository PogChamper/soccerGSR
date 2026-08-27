"""P4: tracklet model over valid fragments -> priority-merge generic jersey cache.

Vote-first hybrid frozen on the dev split (tau 0.10, conf 0.50, override 0.87):
fragments the incumbent generic reader already commits keep their reads; incumbent-
silent fragments where the model commits get its number injected on unread member
boxes; model overrides the incumbent on a fragment only at conf >= 0.87. Output is
jersey_<out-tag>.pkl in the standard generic format, so the assembler needs nothing
new.

    python pix_infer.py --ckpt .../pix_c_s42/best.pt --out-tag pixmix1
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import os
import numpy as np
import torch

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr/jersey_track")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")

from build_bt3_top4_cache import split_tracks, load_pickle  # noqa: E402
from tracklet_pix import TrackletPix, EVAL_TF, SIZE  # noqa: E402

OUT_ROOT = Path("/home/dxdxxd/projects/soccer-app/gsr/out")
CROPS_ZIP = Path("/home/dxdxxd/projects/football/ltpi-research/cache/bt3_sportsl_valid_k16/topk_crops.zip")
OCR_CONF_TH = 0.5
MIN_VOTES = 6
MIN_VOTE_FRAC = 0.02


def fragment_decisions(model, device, seq: str, zf: zipfile.ZipFile, manifest) -> dict:
    """{split_tid: (number, digit_conf, absent_p)} from the top-K crop zip."""
    rows = [r for r in manifest if r["seq"] == seq]
    by_frag = defaultdict(list)
    for r in rows:
        by_frag[int(r["split_tid"])].append(r["name"])
    out = {}
    for tid, names in by_frag.items():
        imgs = []
        for n in names:
            im = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            imgs.append(EVAL_TF(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"])
        if not imgs:
            continue
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].to(device)
        mask = torch.zeros(1, x.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            t, u, a = model(x, mask)[:3]
        pt, pu = torch.softmax(t.float(), 1)[0], torch.softmax(u.float(), 1)[0]
        num = int(pu.argmax()) if int(pt.argmax()) == 0 else int(pt.argmax()) * 10 + int(pu.argmax())
        out[tid] = (num, float(pt.max() * pu.max()), float(torch.sigmoid(a.float())[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seqs", default=",".join(f"SNGS-{i:03d}" for i in range(21, 33)))
    ap.add_argument("--base-tag", default="jnrstar2_u02_c08")
    ap.add_argument("--out-tag", default="pixmix1")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--override-conf", type=float, default=0.87)
    ap.add_argument("--fill-weight", type=float, default=0.6)
    ap.add_argument("--split-eps", type=float, default=0.22)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = TrackletPix().to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()

    zf = zipfile.ZipFile(CROPS_ZIP)
    manifest = list(__import__("csv").DictReader(io.TextIOWrapper(zf.open("manifest.csv"))))

    stats = defaultdict(int)
    for seq in args.seqs.split(","):
        sd = OUT_ROOT / args.split / seq
        track = load_pickle(sd / "track_base.pkl")["records"]
        emb = load_pickle(sd / "emb.pkl")["frames"]
        records = split_tracks(track, emb, args.split_eps)
        jer = load_pickle(sd / f"jersey_{args.base_tag}.pkl")["frames"]
        det = load_pickle(sd / "det.pkl")
        members = defaultdict(list)
        for fi, j, tid, _, _, _ in records:
            members[int(tid)].append((int(fi), int(j)))
        dec = fragment_decisions(model, args.device, seq, zf, manifest)
        dd = Path(os.environ.get("PIX_DEC_DIR", "/mnt/d/jersey-lab/tracklets_v1/decisions"))
        dd.mkdir(parents=True, exist_ok=True)
        json.dump({str(t): d for t, d in dec.items()}, open(dd / f"{seq}.json", "w"))

        nums = [f[0].copy() for f in jer]
        confs = [f[1].copy() for f in jer]
        for tid, mem in members.items():
            reads = [(fi, j) for fi, j in mem
                     if j < len(nums[fi]) and nums[fi][j] >= 0 and confs[fi][j] >= OCR_CONF_TH]
            nv, nb = len(reads), len(mem)
            incumbent_commits = nv >= MIN_VOTES and nv >= MIN_VOTE_FRAC * nb
            d = dec.get(tid)
            if d is None:
                continue
            num, c, ab = d
            model_commits = ab <= args.tau and c >= args.conf
            override = ab <= args.tau and c >= args.override_conf
            if override:
                for fi, j in mem:
                    if j < len(nums[fi]):
                        nums[fi][j] = num
                        confs[fi][j] = 1.0
                stats["override"] += 1
            elif not incumbent_commits and model_commits:
                for fi, j in mem:
                    if j < len(nums[fi]) and nums[fi][j] < 0:
                        nums[fi][j] = num
                        confs[fi][j] = args.fill_weight
                stats["fill"] += 1
            else:
                stats["keep" if incumbent_commits else "silent"] += 1
        out = sd / f"jersey_{args.out_tag}.pkl"
        import pickle
        with open(out, "wb") as f:
            pickle.dump({"frames": [(n, c) for n, c in zip(nums, confs)]}, f)
        print(f"[{seq}] fragments {len(members)} -> {out.name}", flush=True)
    print(f"decisions: {dict(stats)}")


if __name__ == "__main__":
    main()
