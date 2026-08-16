"""Stage 2 (dfine-reid env): OSNet ReID embeddings + PnLCalib homography.

For each Stage-1 detection box we crop and embed with the SoccerNet-finetuned
OSNet (512-d, drives BoT-SORT appearance + team clustering); for each frame we
run the PnLCalib two-HRNet calibrator to get an image->pitch homography (Stage 3
projects box bottom points to pitch metres). cv2 must be <4.11 (calibrateCamera).

    python gsr/s2_embed_calib.py <split> <seq>
"""
import importlib
import os
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research/scripts")
PLUGIN = "/home/dxdxxd/projects/AuxFlow/AuxFlow/plugins/calibration"
sys.path.insert(0, PLUGIN)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

import common as C
from extract_reid_torch import build_osnet

OSNET = "/home/dxdxxd/projects/dataIntegratorSoccer/models/osnet_x1_0_soccernet.pt"
CKP = "/home/dxdxxd/projects/AuxFlow/AuxFlow/pretrained_models/calibration"
DEV = "cuda"

_hm = importlib.import_module("pnlcalib.utils.utils_heatmap")
_calib = importlib.import_module("pnlcalib.utils.utils_calib")
_cfgdir = os.path.join(PLUGIN, "pnlcalib", "config")


def build_calibrator():
    cls_hrnet = importlib.import_module("pnlcalib.model.cls_hrnet")
    cls_hrnet_l = importlib.import_module("pnlcalib.model.cls_hrnet_l")
    cfg_kp = yaml.safe_load(open(os.path.join(_cfgdir, "hrnetv2_w48.yaml")))
    cfg_line = yaml.safe_load(open(os.path.join(_cfgdir, "hrnetv2_w48_l.yaml")))
    mk = cls_hrnet.get_cls_net(cfg_kp)
    mk.load_state_dict(torch.load(CKP + "/SV_kp", map_location=DEV))
    mk.to(DEV).eval()
    ml = cls_hrnet_l.get_cls_net(cfg_line)
    ml.load_state_dict(torch.load(CKP + "/SV_lines", map_location=DEV))
    ml.to(DEV).eval()
    tf = T.Compose([T.Resize((540, 960)), T.ToTensor()])
    return mk, ml, tf


def calibrate(frame_bgr, mk, ml, tf):
    W, H = frame_bgr.shape[1], frame_bgr.shape[0]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = tf(Image.fromarray(rgb)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        hmaps, hmaps_l = mk(t), ml(t)
    kpc = _hm.get_keypoints_from_heatmap_batch_maxpool(hmaps[:, :-1, :, :])
    lnc = _hm.get_keypoints_from_heatmap_batch_maxpool_l(hmaps_l[:, :-1, :, :])
    kpd = _hm.coords_to_dict(kpc, threshold=0.1611, ground_plane_only=True)
    lnd = _hm.coords_to_dict(lnc, threshold=0.3434, ground_plane_only=True)
    kpd, lnd = _hm.complete_keypoints(kpd[0], lnd[0], w=960, h=540, normalize=True)
    cam = _calib.FramebyFrameCalib(iwidth=W, iheight=H, denormalize=True)
    cam.update(kpd, lnd)
    fd = cam.heuristic_voting_ground(refine_lines=False)
    if fd is None:
        return None
    return np.asarray(fd["homography"], np.float64)


def process(split, seq, emb, mk, ml, tf):
    det = C.load(C.OUT_ROOT / split / seq / "det.pkl")
    _, files = C.frame_index(C.load_labels(split, seq))
    sd = C.seq_dir(split, seq)

    emb_frames, calib_frames = [], []
    nfail = 0
    for fi, fname in enumerate(files):
        boxes = det["frames"][fi]
        frame = cv2.imread(str(sd / "img1" / fname))
        if frame is None:
            emb_frames.append(np.zeros((len(boxes), 512), np.float32))
            calib_frames.append(None)
            nfail += 1
            continue
        h, w = frame.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                crops.append(np.zeros((8, 4, 3), np.uint8))
            else:
                crops.append(frame[y1:y2, x1:x2])
        e = emb.embed(crops).astype(np.float32) if crops else np.zeros((0, 512), np.float32)
        emb_frames.append(e)
        try:
            H = calibrate(frame, mk, ml, tf)
        except Exception:
            H = None
        if H is None:
            nfail += 1
        calib_frames.append(H)

    C.dump({"frames": emb_frames}, C.OUT_ROOT / split / seq / "emb.pkl")
    C.dump({"frames": calib_frames}, C.OUT_ROOT / split / seq / "calib.pkl")
    print(f"[s2 {seq}] embedded {sum(len(f) for f in emb_frames)} boxes, "
          f"calib fail {nfail}/{len(files)} -> emb.pkl+calib.pkl", flush=True)


def main():
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    emb = build_osnet(OSNET, device=DEV)
    mk, ml, tf = build_calibrator()
    for seq in seqs:
        process(split, seq, emb, mk, ml, tf)


if __name__ == "__main__":
    main()
