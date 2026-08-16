"""Stage 1b (soccer_eda env): recompute jersey logits from cached boxes + frames.

Decoupled from detection so jersey-reading variants (crop ROI, thresholds) can be
A/B tested without re-detecting. Writes jersey_<tag>.pkl aligned to det.pkl boxes.
The 'torso' ROI focuses the OCR on the number region (upper torso) instead of the
full head-to-toe player box, matching the reader's training distribution.

    python gsr/s1b_jersey.py <split> <seqs|all>
Env: GSR_JERSEY_TAG (output suffix), GSR_ROI = full|torso, plus torso fractions
     GSR_ROI_T/GSR_ROI_B/GSR_ROI_L/GSR_ROI_R (top/bottom/left/right of box height/width).
"""
import os
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
import cv2
import numpy as np
import onnxruntime as ort

import common as C

OCR = "/home/dxdxxd/projects/soccer-app/models/jersey_ocr.onnx"
VIS = "/home/dxdxxd/projects/soccer-app/models/visibility_gate.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

TAG = os.environ.get("GSR_JERSEY_TAG", "torso")
ROI = os.environ.get("GSR_ROI", "torso")
RT = float(os.environ.get("GSR_ROI_T", "0.10"))
RB = float(os.environ.get("GSR_ROI_B", "0.52"))
RL = float(os.environ.get("GSR_ROI_L", "0.08"))
RR = float(os.environ.get("GSR_ROI_R", "0.92"))


def prep(crop_bgr, sz):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    a = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def roi(b, w, h):
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    if ROI == "full":
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
    else:
        bw, bh = x2 - x1, y2 - y1
        rx1, rx2 = x1 + RL * bw, x1 + RR * bw
        ry1, ry2 = y1 + RT * bh, y1 + RB * bh
    return (int(max(0, rx1)), int(max(0, ry1)), int(min(w, rx2)), int(min(h, ry2)))


def process(split, seq, so, sv):
    base = C.OUT_ROOT / split / seq
    det = C.load(base / "det.pkl")
    _, files = C.frame_index(C.load_labels(split, seq))
    sd = C.seq_dir(split, seq)
    jer_frames = []
    for fi, boxes in enumerate(det["frames"]):
        n = len(boxes)
        vis = np.full(n, -1e9, np.float32)
        tens = np.zeros((n, 10), np.float32); units = np.zeros((n, 10), np.float32)
        frame = cv2.imread(str(sd / "img1" / files[fi])) if n else None
        if frame is not None:
            h, w = frame.shape[:2]
            idx, c224, c128 = [], [], []
            for j, b in enumerate(boxes):
                if int(b[5]) not in (0, 1):
                    continue
                rx1, ry1, rx2, ry2 = roi(b, w, h)
                if rx2 - rx1 < 4 or ry2 - ry1 < 6:
                    continue
                crop = frame[ry1:ry2, rx1:rx2]
                idx.append(j); c224.append(prep(crop, 224)); c128.append(prep(crop, 128))
            if idx:
                vlog = sv.run(None, {"input": np.stack(c128).astype(np.float32)})[0].reshape(-1)
                lt, lu = so.run(None, {"input": np.stack(c224).astype(np.float32)})
                for k, j in enumerate(idx):
                    vis[j] = vlog[k]; tens[j] = lt[k]; units[j] = lu[k]
        jer_frames.append((vis, tens, units))
    C.dump({"frames": jer_frames}, base / f"jersey_{TAG}.pkl")
    print(f"[s1b {seq}] roi={ROI} -> jersey_{TAG}.pkl", flush=True)


def main():
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    so = ort.InferenceSession(OCR, providers=prov)
    sv = ort.InferenceSession(VIS, providers=prov)
    for seq in seqs:
        process(split, seq, so, sv)


if __name__ == "__main__":
    main()
