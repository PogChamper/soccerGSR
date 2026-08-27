"""Tracklet reader P3: ConvNeXt-T trunk + cross-crop transformer + tied-digit head.

The pixel arm of the tracklet-reader track. One forward consumes K crops of a GT
track (or fragment) and emits one (tens, units, absent) decision, so sub-threshold
glimpses can be integrated across crops - the information the frozen per-crop
readers discard (P2 showed no feature-level model can recover it). Clean
provenance: ImageNet-init trunk, GSR-train GT tracks + FIFA front-view abstain
crops, selection on the dev split only.

    python tracklet_pix.py train --out /mnt/d/jersey-lab/tracklets_v1/runs/pix_a
    python tracklet_pix.py eval --ckpt .../pix_a/best.pt --split dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

V1 = Path("/mnt/d/jersey-lab/tracklets_v1")
POOL = Path("/mnt/d/jersey-lab/pool")
FIFA_LABELS = Path("/mnt/d/jersey-lab/gsr_abstain/fifa_front_labels")
FIFA_CROPS = Path("/mnt/d/jersey-lab/mine_fifa/crops")
SIZE = 224

TRAIN_TF = A.Compose([
    A.Resize(SIZE, SIZE),
    A.OneOf([A.Downscale(scale_range=(0.25, 0.7)), A.ImageCompression(quality_range=(15, 55))], p=0.4),
    A.Rotate(limit=15, p=0.5),
    A.OneOf([A.MotionBlur(blur_limit=7), A.GaussNoise(std_range=(0.04, 0.12)), A.Defocus(radius=(3, 5))], p=0.4),
    A.OneOf([A.CLAHE(clip_limit=(1, 4)), A.RandomBrightnessContrast(0.2, 0.2), A.RandomGamma((80, 120))], p=0.5),
    A.HueSaturationValue(20, 40, 30, p=0.5),
    A.ToGray(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                    hole_width_range=(0.05, 0.15), p=0.2),
    A.Normalize(),
])
EVAL_TF = A.Compose([A.Resize(SIZE, SIZE), A.Normalize()])


def load_track_table(split: str) -> list[dict]:
    """[{seq, track, label, paths, heights}] for player GT tracks of a split."""
    if split == "train":
        a = pd.read_csv(POOL / "gt_crops_scores_train.csv")
        a = a.rename(columns={"seq_name": "seq"})[["seq", "track_id", "gt_num", "path", "h", "vis_p"]]
        b = pd.read_csv(V1 / "crops_tail17" / "index.csv")
        b = b[b["role"] == "player"].copy()
        b["gt_num"] = pd.to_numeric(b["gt_jersey"], errors="coerce")
        b["vis_p"] = 1.0
        b = b[["seq", "track_id", "gt_num", "path", "h", "vis_p"]]
        df = pd.concat([a, b], ignore_index=True)
    elif split == "dev":
        df = pd.read_csv(POOL / "gt_crops_scores_valid_dev_deployed.csv")
        df = df[df["role"] == "player"].rename(columns={"seq_name": "seq"})
        df = df[["seq", "track_id", "gt_num", "path", "h", "vis_p"]]
    else:
        raise ValueError(split)
    out = []
    for (seq, track), g in df.groupby(["seq", "track_id"]):
        label = g["gt_num"].iloc[0]
        out.append({"seq": seq, "track": int(track),
                    "label": int(label) if pd.notna(label) else -1,
                    "paths": g["path"].tolist(), "heights": g["h"].to_numpy(float),
                    "vis": g["vis_p"].fillna(0.0).to_numpy(float)})
    return out


EXTERNAL = [
    ("/home/dxdxxd/projects/dataIntegratorSoccer/data/jersey-2023/train_refined/images",
     "/home/dxdxxd/projects/dataIntegratorSoccer/data/jersey-2023/train_refined/labels"),
    ("/mnt/d/datasets/soccer_crops/images/player", "/mnt/d/datasets/soccer_crops/labels/player"),
    ("/mnt/d/datasets/soccer_crops_p2/images/player", "/mnt/d/datasets/soccer_crops_p2/labels2/player"),
    ("/mnt/d/datasets/soccer_crops_p3/images/player", "/mnt/d/datasets/soccer_crops_p3/labels2/player"),
    ("/mnt/d/datasets/soccer_crops_p4/images/player", "/mnt/d/datasets/soccer_crops_p4/labels2/player"),
]
READ_OK = {"medium", "high"}


def load_external_crops() -> list[dict]:
    """Known-number external crops as single-crop tracklets (kit diversity)."""
    out = []
    for img_dir, lab_dir in EXTERNAL:
        img_dir, lab_dir = Path(img_dir), Path(lab_dir)
        if not lab_dir.exists():
            continue
        for f in lab_dir.rglob("*.json"):
            try:
                d = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            n = d.get("jersey_number")
            if n is None or not (1 <= int(n) <= 99):
                continue
            if d.get("readability", "medium") not in READ_OK:
                continue
            img = img_dir / f.relative_to(lab_dir).with_suffix(".jpg")
            if img.exists():
                out.append({"seq": "ext", "track": -1, "label": int(n),
                            "paths": [str(img)], "heights": np.array([1.0]),
                            "vis": np.array([1.0])})
    return out


def load_fifa_absent() -> list[dict]:
    out = []
    for f in FIFA_LABELS.glob("*.json"):
        p = FIFA_CROPS / f"{f.stem}.jpg"
        if p.exists():
            out.append({"seq": "fifa", "track": -1, "label": -1,
                        "paths": [str(p)], "heights": np.array([1.0]),
                        "vis": np.array([1.0])})
    return out


class TrackDataset(Dataset):
    def __init__(self, tracks: list[dict], k: int, train: bool, seed: int = 0):
        self.tracks, self.k, self.train = tracks, k, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tracks)

    def pick(self, t: dict) -> tuple[list[str], np.ndarray]:
        n = len(t["paths"])
        vis = t["vis"]
        if self.train:
            # fragment simulation: contiguous window, then top-vis half + random half
            w = int(self.rng.integers(max(1, min(4, n)), n + 1))
            s = int(self.rng.integers(0, n - w + 1))
            idx = np.arange(s, s + w)
            if len(idx) > self.k:
                top = idx[np.argsort(vis[idx])[::-1][: self.k // 2]]
                rest = np.setdiff1d(idx, top)
                rnd = self.rng.choice(rest, self.k - len(top), replace=False)
                idx = np.concatenate([top, rnd])
        else:
            idx = np.argsort(vis)[::-1][: self.k]
        idx = np.sort(idx)
        return [t["paths"][i] for i in idx], vis[idx]

    def __getitem__(self, i: int):
        t = self.tracks[i]
        tf = TRAIN_TF if self.train else EVAL_TF
        paths, vis = self.pick(t)
        imgs, keep = [], []
        for j, p in enumerate(paths):
            im = cv2.imread(p)
            if im is None:
                continue
            imgs.append(tf(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"])
            keep.append(j)
        if not imgs:
            imgs, keep = [np.zeros((SIZE, SIZE, 3), np.float32)], [0]
            vis = np.zeros(1)
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
        v = torch.tensor(vis[keep], dtype=torch.float32)
        label = t["label"]
        return x, v, max(label, 0) // 10, max(label, 0) % 10, float(label < 0)


def collate(batch):
    K = max(x.shape[0] for x, *_ in batch)
    xs = torch.zeros(len(batch), K, 3, SIZE, SIZE)
    vs = torch.zeros(len(batch), K)
    mask = torch.ones(len(batch), K, dtype=torch.bool)
    for i, (x, v, *_ ) in enumerate(batch):
        xs[i, : x.shape[0]] = x
        vs[i, : x.shape[0]] = v
        mask[i, : x.shape[0]] = False
    tens = torch.tensor([t for _, _, t, _, _ in batch])
    units = torch.tensor([u for _, _, _, u, _ in batch])
    absent = torch.tensor([a for _, _, _, _, a in batch], dtype=torch.float32)
    return xs, vs, mask, tens, units, absent


class TrackletPix(nn.Module):
    def __init__(self, d: int = 256, layers: int = 2, heads: int = 8, tied: bool = True):
        super().__init__()
        self.trunk = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.trunk.classifier = nn.Identity()
        self.proj = nn.Linear(768, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.tied = tied
        if tied:
            self.digits = nn.Linear(d, 10, bias=False)
            self.pos = nn.Parameter(torch.randn(2, d) * 0.02)
        else:
            self.head_tens = nn.Linear(d, 10)
            self.head_units = nn.Linear(d, 10)
        self.head_absent = nn.Linear(d, 1)

    def forward(self, x, mask):
        B, K = x.shape[:2]
        f = self.trunk(x.flatten(0, 1)).reshape(B, K, -1)
        h = self.proj(f)
        h = torch.cat([self.cls.expand(B, 1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        tok = self.proj(f)
        h = self.enc(h, src_key_padding_mask=pad)[:, 0]
        if self.tied:
            t = self.digits(h * (1 + self.pos[0]))
            u = self.digits(h * (1 + self.pos[1]))
            tc = self.digits(tok * (1 + self.pos[0]))
            uc = self.digits(tok * (1 + self.pos[1]))
        else:
            t, u = self.head_tens(h), self.head_units(h)
            tc, uc = self.head_tens(tok), self.head_units(tok)
        return t, u, self.head_absent(h)[:, 0], tc, uc


def profile(model, loader, meta, device, taus):
    model.eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
        for xs, _, mask, *_ in loader:
            t, u, a, _, _ = model(xs.to(device), mask.to(device))
            num = torch.where(t.argmax(1) == 0, u.argmax(1), t.argmax(1) * 10 + u.argmax(1))
            outs.append((num.cpu(), torch.sigmoid(a).float().cpu()))
    num = torch.cat([n for n, _ in outs])
    pa = torch.cat([p for _, p in outs])
    best = None
    for tau in taus:
        res = {"hit": 0, "wrong": 0, "miss": 0, "fc": 0, "abstain_ok": 0}
        for t, n, p in zip(meta, num, pa):
            ab = p > tau
            if t["label"] < 0:
                res["fc" if not ab else "abstain_ok"] += 1
            elif ab:
                res["miss"] += 1
            elif int(n) == t["label"]:
                res["hit"] += 1
            else:
                res["wrong"] += 1
        net = res["hit"] - res["wrong"] - res["fc"]
        if best is None or net > best[2]:
            best = (tau, res, net)
    return best


def cmd_train(args):
    torch.manual_seed(args.seed)
    device = args.device
    train_tracks = load_track_table("train")
    if args.external:
        ext = load_external_crops()
        print(f"external known crops: {len(ext)}")
        train_tracks += ext
    if args.fifa:
        fifa = load_fifa_absent()
        rng = np.random.default_rng(args.seed)
        fifa = [fifa[i] for i in rng.choice(len(fifa), min(args.fifa_cap, len(fifa)), replace=False)]
        train_tracks += fifa
        print(f"fifa absent crops: {len(fifa)} (capped)")
    dev_tracks = load_track_table("dev")
    print(f"train tracks {len(train_tracks)}, dev tracks {len(dev_tracks)}")
    train_ds = TrackDataset(train_tracks, args.k, True, args.seed)
    dev_ds = TrackDataset(dev_tracks, args.k_eval, False)
    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          collate_fn=collate, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    dev_ld = DataLoader(dev_ds, args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, persistent_workers=args.workers > 0)
    model = TrackletPix(tied=not args.two_head).to(device)
    groups = {"trunk_w": [], "trunk_b": [], "head_w": [], "head_b": []}
    for n_, p in model.named_parameters():
        k = "trunk" if n_.startswith("trunk.") else "head"
        groups[k + ("_b" if p.ndim <= 1 else "_w")].append(p)
    opt = torch.optim.AdamW([
        {"params": groups["trunk_w"], "lr": args.lr * args.trunk_lr_mult, "weight_decay": 0.05},
        {"params": groups["trunk_b"], "lr": args.lr * args.trunk_lr_mult, "weight_decay": 0.0},
        {"params": groups["head_w"], "lr": args.lr, "weight_decay": 0.05},
        {"params": groups["head_b"], "lr": args.lr, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    la_t = la_u = None
    if args.logit_adjust > 0:
        # balanced-softmax logit adjustment against the head-class prior collapse
        ct, cu = np.ones(10), np.ones(10)
        for t in train_tracks:
            if t["label"] >= 0:
                ct[t["label"] // 10] += 1
                cu[t["label"] % 10] += 1
        la_t = torch.log(torch.tensor(ct / ct.sum(), dtype=torch.float32)).to(device) * args.logit_adjust
        la_u = torch.log(torch.tensor(cu / cu.sum(), dtype=torch.float32)).to(device) * args.logit_adjust
    n_pos = sum(1 for t in train_tracks if t["label"] < 0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_tracks) - n_pos) / max(n_pos, 1)).to(device))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taus = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_net = -10**9
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for xs, vs, mask, tens, units, absent in train_ld:
            xs, mask = xs.to(device, non_blocking=True), mask.to(device)
            vs = vs.to(device)
            with torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                t, u, a, tc, uc = model(xs, mask)
                known = absent.to(device) < 0.5
                loss = bce(a, absent.to(device))
                if known.any():
                    tl = t[known] + la_t if la_t is not None else t[known]
                    ul = u[known] + la_u if la_u is not None else u[known]
                    loss = loss + ce(tl, tens.to(device)[known]) + ce(ul, units.to(device)[known])
                    # per-crop supervision on trunk tokens: the track label broadcast
                    # to every real crop of known tracks (kit-diverse crops included)
                    km = known[:, None] & ~mask & (vs >= 0.7)
                    if km.any():
                        tcl = tc[km] + la_t if la_t is not None else tc[km]
                        ucl = uc[km] + la_u if la_u is not None else uc[km]
                        loss = loss + args.aux_w * (
                            ce(tcl, tens.to(device)[:, None].expand_as(km)[km]) +
                            ce(ucl, units.to(device)[:, None].expand_as(km)[km]))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
        sched.step()
        msg = f"ep {ep} loss {np.mean(losses):.4f} ({time.time() - t0:.0f} s)"
        if ep >= args.eval_from and (ep % args.eval_every == 0 or ep == args.epochs - 1):
            tau, p, net = profile(model, dev_ld, dev_tracks, device, taus)
            msg += f" | dev tau {tau}: {p} net {net}"
            if net > best_net:
                best_net = net
                torch.save({"model": model.state_dict(), "tau": tau, "epoch": ep,
                            "dev_profile": p, "args": vars(args)}, out / "best.pt")
                msg += " *"
        print(msg, flush=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1, "args": vars(args)},
               out / "last.pt")
    print(f"done, best dev net {best_net}, ckpt {out}/best.pt")


def cmd_eval(args):
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    a = ck.get("args", {})
    model = TrackletPix(tied=not a.get("two_head", False)).to(args.device)
    model.load_state_dict(ck["model"])
    tracks = load_track_table(args.split)
    ds = TrackDataset(tracks, a.get("k_eval", 16), False)
    ld = DataLoader(ds, 16, shuffle=False, num_workers=8, collate_fn=collate)
    taus = [ck["tau"]] if args.locked_tau else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tau, p, net = profile(model, ld, tracks, args.device, taus)
    known = sum(1 for t in tracks if t["label"] >= 0)
    print(f"{args.split} tau {tau}: known {known} -> {p} net {net}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out", required=True)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--k", type=int, default=12)
    tr.add_argument("--k-eval", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--workers", type=int, default=8)
    tr.add_argument("--fifa", action="store_true", default=True)
    tr.add_argument("--no-fifa", dest="fifa", action="store_false")
    tr.add_argument("--fifa-cap", type=int, default=800)
    tr.add_argument("--external", action="store_true", default=True)
    tr.add_argument("--no-external", dest="external", action="store_false")
    tr.add_argument("--aux-w", type=float, default=1.0)
    tr.add_argument("--trunk-lr-mult", type=float, default=0.1)
    tr.add_argument("--logit-adjust", type=float, default=0.0)
    tr.add_argument("--two-head", action="store_true")
    tr.add_argument("--eval-from", type=int, default=2)
    tr.add_argument("--eval-every", type=int, default=2)
    tr.add_argument("--device", default="cuda")
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--split", default="dev")
    ev.add_argument("--locked-tau", action="store_true")
    ev.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
