"""Tracklet reader v0: a small transformer over full-rate per-detection reader logits.

The control arm of the tracklet-reader track: consumes the frozen ConvNeXt two-head
logits plus box geometry for every detection of a track and emits one (tens, units,
absent) decision. Trains on GSR-train GT tracks (933 known / 212 None players / 79 GK
as absent), with fragment simulation (random contiguous windows + crop dropout).
Selection happens on the dev split (valid games 3+5), never on valid-12.

    python tracklet_v0.py train --out /mnt/d/jersey-lab/tracklets_v1/runs/v0a
    python tracklet_v0.py eval  --ckpt .../v0a/best.pt --logits-dir .../logits_valid12
    python tracklet_v0.py baselines --logits-dir .../logits_valid12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

V1 = Path("/mnt/d/jersey-lab/tracklets_v1")
FEAT_DIM = 29
MAX_LEN = 128


def softmax_np(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def track_features(z: dict, s: int, e: int) -> np.ndarray:
    """Per-detection feature block for one track slice, float32 (n, FEAT_DIM)."""
    tens = softmax_np(z["m_tens"][s:e])
    units = softmax_np(z["m_units"][s:e])
    vis = 1.0 / (1.0 + np.exp(-z["m_vis"][s:e]))
    conf = np.minimum(tens.max(-1), units.max(-1))
    xyxy = z["m_box"][s:e]
    h = xyxy[:, 3] - xyxy[:, 1]
    w = xyxy[:, 2] - xyxy[:, 0]
    geo = np.stack([np.log(np.maximum(h, 1.0)) / 6.0, np.log(np.maximum(w, 1.0)) / 6.0,
                    w / np.maximum(h, 1.0), z["m_iou"][s:e], z["m_conf"][s:e]], axis=1)
    # the incumbent's information: the shipped gate decision per crop, and its track rate
    accepted = ((vis >= 0.7) & (conf >= 0.95)).astype(np.float32)
    acc_frac = np.full_like(accepted, accepted.mean())
    return np.concatenate([tens, units, vis[:, None], conf[:, None], geo,
                           accepted[:, None], acc_frac[:, None]], axis=1).astype(np.float32)


def load_tracks(logits_dir: Path):
    """[(seq, track, role, label, features)] for every player/gk GT track."""
    out = []
    for f in sorted(logits_dir.glob("SNGS-*.npz")):
        z = dict(np.load(f))
        for track, role, label, s, e in zip(z["t_track"], z["t_role"], z["t_label"],
                                            z["t_start"], z["t_end"]):
            if e <= s:
                continue
            out.append((f.stem, int(track), int(role), int(label),
                        track_features(z, int(s), int(e))))
    return out


class TrackletV0(nn.Module):
    def __init__(self, d: int = 96, layers: int = 2, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head_tens = nn.Linear(d, 10)
        self.head_units = nn.Linear(d, 10)
        self.head_absent = nn.Linear(d, 1)

    def forward(self, x, mask):
        # x (B, L, FEAT_DIM), mask True on padding
        h = self.proj(x)
        h = torch.cat([self.cls.expand(len(h), 1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), mask], dim=1)
        h = self.enc(h, src_key_padding_mask=pad)[:, 0]
        return self.head_tens(h), self.head_units(h), self.head_absent(h)[:, 0]


def sample_window(feat: np.ndarray, train: bool, rng: np.random.Generator) -> np.ndarray:
    n = len(feat)
    if train:
        w = int(rng.integers(min(16, n), n + 1))
        s = int(rng.integers(0, n - w + 1))
        feat = feat[s:s + w]
        if len(feat) > 4:
            keep = rng.random(len(feat)) > rng.uniform(0.0, 0.5)
            if keep.sum() >= 4:
                feat = feat[keep]
    if len(feat) > MAX_LEN:
        idx = np.linspace(0, len(feat) - 1, MAX_LEN).astype(int)
        feat = feat[idx]
    return feat


def collate(batch):
    L = max(len(f) for f, *_ in batch)
    x = torch.zeros(len(batch), L, FEAT_DIM)
    mask = torch.ones(len(batch), L, dtype=torch.bool)
    for i, (f, *_ ) in enumerate(batch):
        x[i, : len(f)] = torch.from_numpy(f)
        mask[i, : len(f)] = False
    tens = torch.tensor([t for _, t, _, _ in batch])
    units = torch.tensor([u for _, _, u, _ in batch])
    absent = torch.tensor([a for _, _, _, a in batch], dtype=torch.float32)
    return x, mask, tens, units, absent


def profile(model, tracks, device, tau: float, batch: int = 256):
    """hit / wrong / miss on known tracks, false commits + correct abstains on None."""
    model.eval()
    res = {"hit": 0, "wrong": 0, "miss": 0, "fc": 0, "abstain_ok": 0}
    with torch.no_grad():
        for i in range(0, len(tracks), batch):
            chunk = tracks[i:i + batch]
            feats = [sample_window(f, False, None) for *_, f in chunk]
            L = max(len(f) for f in feats)
            x = torch.zeros(len(chunk), L, FEAT_DIM)
            mask = torch.ones(len(chunk), L, dtype=torch.bool)
            for j, f in enumerate(feats):
                x[j, : len(f)] = torch.from_numpy(f)
                mask[j, : len(f)] = False
            t, u, a = model(x.to(device), mask.to(device))
            num = torch.where(t.argmax(1) == 0, u.argmax(1), t.argmax(1) * 10 + u.argmax(1))
            abstain = torch.sigmoid(a) > tau
            for (seq, track, role, label, _), n, ab in zip(chunk, num.cpu(), abstain.cpu()):
                if label < 0:
                    res["fc" if not ab else "abstain_ok"] += 1
                elif ab:
                    res["miss"] += 1
                elif int(n) == label:
                    res["hit"] += 1
                else:
                    res["wrong"] += 1
    return res


def net(p) -> int:
    return p["hit"] - p["wrong"] - p["fc"]


def cmd_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    train = load_tracks(V1 / "logits_train")
    dev = load_tracks(Path(args.dev_dir)) if args.dev_dir else None
    print(f"train tracks {len(train)}, dev tracks {len(dev) if dev else 0}")
    model = TrackletV0(args.dim, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(933 / 291).to(device))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -10**9
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for i in range(0, len(order), args.batch):
            items = []
            for k in order[i:i + args.batch]:
                seq, track, role, label, feat = train[k]
                f = sample_window(feat, True, rng)
                items.append((f, max(label, 0) // 10, max(label, 0) % 10, float(label < 0)))
            x, mask, tens, units, absent = collate(items)
            x, mask = x.to(device), mask.to(device)
            t, u, a = model(x, mask)
            known = absent.to(device) < 0.5
            loss = bce(a, absent.to(device))
            if known.any():
                loss = loss + ce(t[known], tens.to(device)[known]) + ce(u[known], units.to(device)[known])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        msg = f"ep {ep} loss {np.mean(losses):.4f}"
        if dev:
            taus = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            profs = {tau: profile(model, dev, device, tau) for tau in taus}
            tau, p = max(profs.items(), key=lambda kv: net(kv[1]))
            msg += f" | dev tau {tau}: {p} net {net(p)}"
            if net(p) > best:
                best = net(p)
                torch.save({"model": model.state_dict(), "tau": tau, "epoch": ep,
                            "dev_profile": p, "args": vars(args)}, out / "best.pt")
                msg += " *"
        print(msg, flush=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1,
                "args": vars(args)}, out / "last.pt")
    print(f"done, best dev net {best}, ckpt {out}/best.pt")


def cmd_eval(args):
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    a = ck.get("args", {})
    model = TrackletV0(a.get("dim", 96), a.get("layers", 2)).to(args.device)
    model.load_state_dict(ck["model"])
    tracks = load_tracks(Path(args.logits_dir))
    tau = args.tau if args.tau is not None else ck.get("tau", 0.5)
    p = profile(model, tracks, args.device, tau)
    known = sum(1 for *_, l, _ in [(s, t, r, l, f) for s, t, r, l, f in tracks] if l >= 0)
    print(f"{args.logits_dir} tau {tau}: known {known} -> {p} net {net(p)}")


def cmd_baselines(args):
    """Parameter-free incumbents on the same tracks: s3b vote and three poolings."""
    tracks = load_tracks(Path(args.logits_dir))
    rules = {}
    for name in ("vote", "meanprob", "logitsum", "certmean"):
        res = {"hit": 0, "wrong": 0, "miss": 0, "fc": 0, "abstain_ok": 0}
        for seq, track, role, label, f in tracks:
            tens, units = f[:, :10], f[:, 10:20]
            vis, conf = f[:, 20], f[:, 21]
            if name == "vote":
                m = (vis >= 0.7) & (conf >= 0.95)
                commit = m.sum() >= 6
                if commit:
                    pt, pu = tens[m].sum(0), units[m].sum(0)
            else:
                w = {"meanprob": np.ones(len(f)), "logitsum": np.ones(len(f)),
                     "certmean": conf}[name]
                pt = (np.log(np.clip(tens, 1e-9, 1)) if name == "logitsum" else tens)
                pu = (np.log(np.clip(units, 1e-9, 1)) if name == "logitsum" else units)
                pt, pu = (pt * w[:, None]).sum(0), (pu * w[:, None]).sum(0)
                commit = True
            if not commit:
                res["miss" if label >= 0 else "abstain_ok"] += 1
                continue
            n = int(pu.argmax()) if pt.argmax() == 0 else int(pt.argmax() * 10 + pu.argmax())
            if label < 0:
                res["fc"] += 1
            elif n == label:
                res["hit"] += 1
            else:
                res["wrong"] += 1
        rules[name] = res
        print(f"{name:9s} {res} net {net(res)}")
    return rules


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out", required=True)
    tr.add_argument("--dev-dir", default=str(V1 / "logits_dev40"))
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--dim", type=int, default=96)
    tr.add_argument("--layers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--device", default="cuda")
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", required=True)
    ev.add_argument("--logits-dir", required=True)
    ev.add_argument("--tau", type=float, default=None)
    ev.add_argument("--device", default="cuda")
    bl = sub.add_parser("baselines")
    bl.add_argument("--logits-dir", required=True)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
