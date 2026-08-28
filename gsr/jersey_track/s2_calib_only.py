"""PnLCalib homographies only (emb.pkl already built) for the dev-40 chain legs.

Reuses s2_embed_calib's calibrator wholesale; writes calib.pkl in the standard
format. Camera source moves the assembler by +-0.3 HOTA only (FIFA measurement),
which is enough for an identity-knob transfer check. Resumable per clip.

    python s2_calib_only.py valid SNGS-039,SNGS-040
"""
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")

import cv2
import numpy as np

import common as C
import s2_embed_calib as S2


def process(split: str, seq: str, mk, ml, tf) -> None:
    out = C.OUT_ROOT / split / seq / "calib.pkl"
    if out.exists():
        print(f"[s2c {seq}] exists, skip", flush=True)
        return
    det = C.load(C.OUT_ROOT / split / seq / "det.pkl")
    sd = C.seq_dir(split, seq)
    frames, nfail = [], 0
    for fi in range(len(det["frames"])):
        frame = cv2.imread(str(sd / "img1" / f"{fi + 1:06d}.jpg"))
        try:
            H = S2.calibrate(frame, mk, ml, tf)
        except Exception:
            H = None
        if H is None:
            nfail += 1
        frames.append(H)
    C.dump({"frames": frames}, out)
    print(f"[s2c {seq}] calib fail {nfail}/{len(frames)} -> calib.pkl", flush=True)


def main() -> None:
    split, seqs = sys.argv[1], sys.argv[2].split(",")
    mk, ml, tf = S2.build_calibrator()
    for seq in seqs:
        process(split, seq, mk, ml, tf)


if __name__ == "__main__":
    main()
