"""Stage 3 (soccer_eda env): associate + assemble -> GS-HOTA predictions JSON.

Runs vendored BoT-SORT (motion + external OSNet appearance) over Stage-1/2
caches, then per tracklet decides role (majority class, which also fixes the
detector's goalkeeper->player folding), team (KMeans(2) on appearance, sides
labelled by mean pitch-x per the reference heuristic, GK by own pitch-x sign),
and jersey (visibility-gated logits-sum vote). Every kept box is projected to
pitch metres via the (hold-last) homography and written in the evaluator schema.

    python gsr/s3_assemble.py <split> <seq> [--relink]
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")

import json

import numpy as np
from sklearn.cluster import KMeans

import os

import common as C
from app.services.detector import Detection
from app.services.tracker import BoxmotTracker

VIS_TH = float(os.environ.get("GSR_VIS_TH", "0.6"))
OCR_CONF_TH = float(os.environ.get("GSR_OCR_CONF_TH", "0.7"))
MIN_VOTES = int(os.environ.get("GSR_MIN_VOTES", "4"))
TAG = os.environ.get("GSR_TAG", "ours")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def fill_homographies(calib):
    """Hold-last (then back-fill leading gaps) so every frame has a homography."""
    H = list(calib)
    last = None
    for i in range(len(H)):
        if H[i] is not None:
            last = H[i]
        elif last is not None:
            H[i] = last
    nxt = None
    for i in range(len(H) - 1, -1, -1):
        if H[i] is not None:
            nxt = H[i]
        elif nxt is not None:
            H[i] = nxt
    return H


def project(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def process(split, seq, do_relink=False):
    base = C.OUT_ROOT / split / seq
    det = C.load(base / "det.pkl")
    jer = C.load(base / "jersey.pkl")
    emb = C.load(base / "emb.pkl")["frames"]
    calib = C.load(base / "calib.pkl")["frames"]
    labels = C.load_labels(split, seq)
    _, files = C.frame_index(labels)
    sd = C.seq_dir(split, seq)
    image_ids = det["image_ids"]
    H_eff = fill_homographies(calib)
    if all(h is None for h in H_eff):
        print(f"[s3 {seq}] no homography at all -> SKIP", flush=True)
        return

    import cv2

    trk = BoxmotTracker(frame_rate=25, with_reid=True)
    # per detection: (frame, box) -> track id ; plus records for aggregation
    records = []  # (fi, j, tid, cls, bbox, conf)
    for fi, boxes in enumerate(det["frames"]):
        frame = cv2.imread(str(sd / "img1" / files[fi]))
        if frame is None:
            frame = np.zeros((1080, 1920, 3), np.uint8)
        dets = [Detection(bbox=(b[0], b[1], b[2], b[3]), class_id=int(b[5]),
                          class_name="", confidence=float(b[4])) for b in boxes]
        e = emb[fi] if len(emb[fi]) == len(dets) else None
        pairs = trk.update(dets, frame, embeddings=e)
        for j, (d, tid) in enumerate(pairs):
            if tid is None:
                continue
            records.append((fi, j, int(tid), int(boxes[j][5]), boxes[j][:4], float(boxes[j][4])))

    if not records:
        print(f"[s3 {seq}] no tracked boxes -> SKIP", flush=True)
        return

    by_track = defaultdict(list)
    for r in records:
        by_track[r[2]].append(r)

    # --- per-track role, mean embedding, jersey vote, pitch-x samples ---
    role_of, emb_of, jersey_of, pitchx_of = {}, {}, {}, {}
    for tid, rs in by_track.items():
        cls_maj = Counter(r[3] for r in rs).most_common(1)[0][0]
        role_of[tid] = C.CLS_TO_ROLE[cls_maj]
        vecs = [emb[fi][j] for fi, j, *_ in rs if len(emb[fi]) > j]
        emb_of[tid] = (np.mean(vecs, 0) if vecs else np.zeros(512, np.float32))
        xs = []
        for fi, j, _, _, bbox, _ in rs:
            xc, yb = (bbox[0] + bbox[2]) / 2.0, bbox[3]
            xs.append(project(H_eff[fi], xc, yb)[0])
        pitchx_of[tid] = float(np.median(xs)) if xs else 0.0

        jersey_of[tid] = None
        if role_of[tid] == "player":
            st, su, nv = np.zeros(10), np.zeros(10), 0
            for fi, j, *_ in rs:
                vis, tens, units = jer["frames"][fi]
                if j >= len(vis) or sigmoid(vis[j]) < VIS_TH:
                    continue
                if min(softmax(tens[j]).max(), softmax(units[j]).max()) < OCR_CONF_TH:
                    continue
                st += tens[j]; su += units[j]; nv += 1
            if nv >= MIN_VOTES:
                t, u = int(np.argmax(st)), int(np.argmax(su))
                jersey_of[tid] = u if t == 0 else t * 10 + u

    # --- team: KMeans(2) on player+gk mean embeddings, sides by mean pitch-x ---
    team_of = {}
    field = [t for t in by_track if role_of[t] in ("player", "goalkeeper")]
    if len(field) >= 2:
        X = np.stack([emb_of[t] / max(np.linalg.norm(emb_of[t]), 1e-9) for t in field])
        lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        cx = {c: np.mean([pitchx_of[t] for t, l in zip(field, lab) if l == c]) for c in (0, 1)}
        left_cluster = 0 if cx[0] <= cx[1] else 1
        for t, l in zip(field, lab):
            team_of[t] = "left" if l == left_cluster else "right"
    else:
        for t in field:
            team_of[t] = "left"
    for t in field:  # GK by own pitch-x sign (reference heuristic)
        if role_of[t] == "goalkeeper":
            team_of[t] = "right" if pitchx_of[t] > 0 else "left"

    # --- write predictions ---
    preds = []
    for fi, j, tid, cls, bbox, conf in records:
        role = role_of[tid]
        cat = {"player": 1, "goalkeeper": 2, "referee": 3}[role]
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        H = H_eff[fi]
        bl, bm, br = project(H, x1, y2), project(H, (x1 + x2) / 2.0, y2), project(H, x2, y2)
        jn = jersey_of[tid] if role == "player" else None
        team = team_of.get(tid) if role in ("player", "goalkeeper") else None
        preds.append({
            "image_id": image_ids[fi],
            "track_id": int(tid),
            "supercategory": "object",
            "category_id": cat,
            "confidence": conf,
            "bbox_pitch": {
                "x_bottom_left": bl[0], "y_bottom_left": bl[1],
                "x_bottom_middle": bm[0], "y_bottom_middle": bm[1],
                "x_bottom_right": br[0], "y_bottom_right": br[1],
            },
            "attributes": {"role": role, "team": team,
                           "jersey": (str(jn) if jn is not None else None)},
        })

    out = {"images": labels["images"], "categories": labels["categories"], "predictions": preds}
    dst = C.OUT_ROOT / "preds" / f"SoccerNetGS-{split}" / TAG / "data" / f"{seq}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"))
    njer = sum(1 for t in by_track if jersey_of.get(t) is not None)
    print(f"[s3 {seq}] {len(by_track)} tracks ({len(field)} field), {njer} with jersey, "
          f"{len(preds)} preds -> {dst}", flush=True)


def main():
    split = sys.argv[1]
    do_relink = "--relink" in sys.argv
    arg = sys.argv[2]
    seqs = C.list_seqs(split) if arg == "all" else arg.split(",")
    for seq in seqs:
        process(split, seq, do_relink)


if __name__ == "__main__":
    main()
