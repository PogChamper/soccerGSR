"""Stage 1 (soccer_eda env): detect + jersey OCR in a single pass over frames.

DEIMv2 supplies boxes with role classes (0=player,1=gk,2=ref,3=ball); ball is
dropped (GS-HOTA ignores it). For every player/gk box the ShuffleNet visibility
gate + ConvNeXt two-head OCR logits are stored per detection so Stage 3 can vote
one jersey per tracklet. Outputs det.pkl (boxes) and jersey.pkl (per-box logits),
both aligned by (frame index, detection index).

    MODEL_AUTO_DOWNLOAD=false python gsr/s1_detect_jersey.py <split> <seq>
"""
import os
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
os.environ.setdefault("MODEL_AUTO_DOWNLOAD", "false")

import cv2
import numpy as np
import onnxruntime as ort

import common as C
from app.utils import cuda_env

cuda_env.bootstrap()
from app.services.detector import DEIMv2Detector

DET = "/home/dxdxxd/projects/soccer-app/models/deimv2_m_896.onnx"
OCR = "/home/dxdxxd/projects/soccer-app/models/jersey_ocr.onnx"
VIS = "/home/dxdxxd/projects/soccer-app/models/visibility_gate.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep(crop_bgr, sz):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    a = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def process(split, seq, det, so, sv):
    labels = C.load_labels(split, seq)
    image_ids, files = C.frame_index(labels)
    sd = C.seq_dir(split, seq)

    det_frames, jer_frames = [], []
    for fi, fname in enumerate(files):
        frame = cv2.imread(str(sd / "img1" / fname))
        if frame is None:
            det_frames.append(np.zeros((0, 6), np.float32))
            jer_frames.append((np.zeros(0, np.float32), np.zeros((0, 10), np.float32), np.zeros((0, 10), np.float32)))
            continue
        h, w = frame.shape[:2]
        boxes = []
        for r in det.detect(frame):
            if r.class_id == 3:  # ball ignored by GS-HOTA
                continue
            boxes.append((r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3], r.confidence, r.class_id))
        arr = np.array(boxes, np.float32) if boxes else np.zeros((0, 6), np.float32)
        det_frames.append(arr)

        n = len(arr)
        vis = np.full(n, -1e9, np.float32)
        tens = np.zeros((n, 10), np.float32)
        units = np.zeros((n, 10), np.float32)
        idx, crops224, crops128 = [], [], []
        for j, b in enumerate(arr):
            if int(b[5]) not in (0, 1):  # only player/gk get OCR
                continue
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            idx.append(j)
            crops224.append(prep(crop, 224))
            crops128.append(prep(crop, 128))
        if idx:
            vlog = sv.run(None, {"input": np.stack(crops128).astype(np.float32)})[0].reshape(-1)
            lt, lu = so.run(None, {"input": np.stack(crops224).astype(np.float32)})
            for k, j in enumerate(idx):
                vis[j] = vlog[k]
                tens[j] = lt[k]
                units[j] = lu[k]
        jer_frames.append((vis, tens, units))

    C.dump({"split": split, "seq": seq, "image_ids": image_ids, "frames": det_frames},
           C.OUT_ROOT / split / seq / "det.pkl")
    C.dump({"frames": jer_frames}, C.OUT_ROOT / split / seq / "jersey.pkl")
    total = sum(len(f) for f in det_frames)
    print(f"[s1 {seq}] {len(files)} frames, {total} person-boxes -> det.pkl+jersey.pkl", flush=True)


def main():
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    det = DEIMv2Detector(model_path=DET)
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    so = ort.InferenceSession(OCR, providers=prov)
    sv = ort.InferenceSession(VIS, providers=prov)
    for seq in seqs:
        process(split, seq, det, so, sv)


if __name__ == "__main__":
    main()
