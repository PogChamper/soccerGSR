"""Star-v2 confirmation filter over the mined SoccerFactory crops.

Keeps a crop when the star v2 reader reads the SAME number as the pseudo-label
at conf >= 0.5 (audit-verified: A+B 91.4 percent, D 0.0 on the kept stream).
CPU ORT; resumable per clip-manifest.

    python sf_confirm.py [--workers 6]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MINE = Path("/mnt/d/jersey-lab/sf_mine_v1")
OUT = MINE / "confirmed"


def make_reader():
    sess = ort.InferenceSession("/mnt/d/jersey-lab/runs/star_v2/reader.onnx",
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def read(img):
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
        x = (x - mean) / std
        t, u = sess.run(None, {inp: x.transpose(2, 0, 1)[None]})
        t, u = t[0], u[0]
        st, su = np.exp(t - t.max()), np.exp(u - u.max())
        st, su = st / st.sum(), su / su.sum()
        ti, ui = int(st.argmax()), int(su.argmax())
        if ti == 10 or ui == 10:
            return None, 0.0
        return (ui if ti == 0 else ti * 10 + ui), float(min(st.max(), su.max()))

    return read


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    read = make_reader()
    for mf in sorted((MINE / "manifests").glob("SNGS-*.csv")):
        of = OUT / mf.name
        if of.exists():
            continue
        kept = []
        for r in csv.DictReader(open(mf)):
            img = cv2.imread(str(MINE / "crops" / r["name"]))
            if img is None:
                continue
            num, conf = read(img)
            if num is not None and str(num) == r["label"] and conf >= args.conf:
                kept.append(r)
        with open(of, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "clip", "frame", "label", "h"])
            w.writeheader()
            w.writerows(kept)
        print(f"[{mf.stem}] kept {len(kept)}", flush=True)


if __name__ == "__main__":
    main()
