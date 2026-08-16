"""Stage 1c (centroids env): PARSeq jersey reader on cached det boxes + frames.

Prototype of the SoccerNet-2024 winning STR recipe as a drop-in stronger reader:
legibility filter (ResNet34) + torso ROI + PARSeq scene-text recognition. Decoupled
from detection like s1b so it A/B tests against the ConvNeXt two-head OCR without
re-detecting. Writes jersey_parseq.pkl aligned to det.pkl boxes.

    conda run -n centroids python gsr/s1c_parseq.py <split> <seqs|all>

Output: out/<split>/<SEQ>/jersey_parseq.pkl =
    {"frames": [ (nums, confs) per frame ]}
  nums  : np.int16   len N (aligned to det.pkl boxes); predicted 0-99, or -1 if
          illegible / no-number / not a cls in {0,1} box.
  confs : np.float32 len N in [0,1]; PARSeq char-prob product, 0.0 where nums=-1.

The reader runs only on player/goalkeeper boxes (cls in {0,1}). A box is committed
only if the legibility classifier passes (>= GSR_LEG_TH) AND PARSeq yields a valid
1-2 digit number; otherwise nums=-1 (abstain). Torso ROI (fractions of the box)
matches PARSeq's ViTPose shoulder-to-hip training crops.

Env (all optional):
  GSR_LEG_TH   legibility gate threshold (default 0.5)
  GSR_ROI      full|torso (default torso)
  GSR_ROI_T/B/L/R  torso fractions of box h/w (default 0.08/0.60/0.02/0.98)
  GSR_PARSEQ_BS    max STR/leg batch size (default 128)
"""
import os
import sys
import time

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/jersey/jersey-number-pipeline/str/parseq")
sys.path.insert(0, "/home/dxdxxd/projects/jersey/jersey-number-pipeline")
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import common as C
from strhub.models.utils import load_from_checkpoint
from networks import LegibilityClassifier34

JP_ROOT = "/home/dxdxxd/projects/jersey/jersey-number-pipeline"
CKPT = f"{JP_ROOT}/models/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
LEG_CKPT = f"{JP_ROOT}/models/legibility_resnet34_soccer_20240215.pth"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

LEG_TH = float(os.environ.get("GSR_LEG_TH", "0.5"))
ROI = os.environ.get("GSR_ROI", "torso")
RT = float(os.environ.get("GSR_ROI_T", "0.08"))
RB = float(os.environ.get("GSR_ROI_B", "0.60"))
RL = float(os.environ.get("GSR_ROI_L", "0.02"))
RR = float(os.environ.get("GSR_ROI_R", "0.98"))
BS = int(os.environ.get("GSR_PARSEQ_BS", "128"))

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]
_parseq_tf = T.Compose([T.Resize([32, 128], T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])
_leg_tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(IN_MEAN, IN_STD)])


def roi_box(b, w, h):
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
    if ROI == "full":
        return int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    bw, bh = x2 - x1, y2 - y1
    rx1, rx2 = x1 + RL * bw, x1 + RR * bw
    ry1, ry2 = y1 + RT * bh, y1 + RB * bh
    return int(max(0, rx1)), int(max(0, ry1)), int(min(w, rx2)), int(min(h, ry2))


def valid_num(s):
    if not s or len(s) > 2:
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    return n if 0 <= n <= 99 else None


def load_models():
    model = load_from_checkpoint(CKPT, charset_test="0123456789").eval().to(DEV)
    lg = LegibilityClassifier34()
    sd = torch.load(LEG_CKPT, map_location=DEV)
    if hasattr(sd, "_metadata"):
        del sd._metadata
    lg.load_state_dict(sd)
    lg = lg.to(DEV).eval()
    return model, lg


@torch.inference_mode()
def parseq_read(model, pil_list):
    """Return list of (num_str, char_prob_product) for each crop."""
    if not pil_list:
        return []
    x = torch.stack([_parseq_tf(im) for im in pil_list]).to(DEV)
    out = []
    for i in range(0, len(x), BS):
        logits = model(x[i:i + BS])
        probs = logits[:, :3, :11].softmax(-1)  # digit-only (E,0..9), 3 positions
        preds, confs = model.tokenizer.decode(probs)
        for p, c in zip(preds, confs):
            cc = c.detach().cpu().numpy().tolist()
            tot = 1.0
            for v in cc[:-1]:  # drop EOS-token prob (matches winning recipe)
                tot *= float(v)
            out.append((p, tot))
    return out


@torch.inference_mode()
def leg_prob(model, pil_list):
    if not pil_list:
        return []
    x = torch.stack([_leg_tf(im) for im in pil_list]).to(DEV)
    out = []
    for i in range(0, len(x), BS):
        out += model(x[i:i + BS]).reshape(-1).detach().cpu().numpy().tolist()
    return out


def process(split, seq, model, lg):
    base = C.OUT_ROOT / split / seq
    det = C.load(base / "det.pkl")
    labels = C.load_labels(split, seq)
    image_ids, files = C.frame_index(labels)
    id2file = dict(zip(image_ids, files))
    sd = C.seq_dir(split, seq)

    jer_frames = []
    t0 = time.time()
    for fi, boxes in enumerate(det["frames"]):
        n = len(boxes)
        nums = np.full(n, -1, np.int16)
        confs = np.zeros(n, np.float32)
        if n:
            img = cv2.imread(str(sd / "img1" / id2file[det["image_ids"][fi]]))
            if img is not None:
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                idx, full_crops, torso_crops = [], [], []
                for j, b in enumerate(boxes):
                    if int(b[5]) not in (0, 1):
                        continue
                    fx1, fy1, fx2, fy2 = roi_box(b, w, h) if ROI == "full" else (
                        int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3])))
                    tx1, ty1, tx2, ty2 = roi_box(b, w, h)
                    if fx2 - fx1 < 3 or fy2 - fy1 < 4 or tx2 - tx1 < 3 or ty2 - ty1 < 4:
                        continue
                    idx.append(j)
                    full_crops.append(Image.fromarray(rgb[fy1:fy2, fx1:fx2]))
                    torso_crops.append(Image.fromarray(rgb[ty1:ty2, tx1:tx2]))
                if idx:
                    legs = leg_prob(lg, full_crops)
                    reads = parseq_read(model, torso_crops)
                    for k, j in enumerate(idx):
                        if legs[k] < LEG_TH:
                            continue
                        num = valid_num(reads[k][0])
                        if num is None:
                            continue
                        nums[j] = num
                        confs[j] = min(1.0, max(0.0, reads[k][1]))
        jer_frames.append((nums, confs))
    C.dump({"frames": jer_frames}, base / "jersey_parseq.pkl")
    dt = time.time() - t0
    ncommit = int(sum((f[0] >= 0).sum() for f in jer_frames))
    print(f"[s1c {seq}] roi={ROI} leg_th={LEG_TH} -> jersey_parseq.pkl "
          f"({len(jer_frames)} frames, {ncommit} commits, {dt:.1f}s, "
          f"peakGPU={torch.cuda.max_memory_allocated()/1e6:.0f}MB)", flush=True)


def main():
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    model, lg = load_models()
    for seq in seqs:
        process(split, seq, model, lg)


if __name__ == "__main__":
    main()
