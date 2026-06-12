"""Export PnLCalib HRNet weights (.pt) -> ONNX.

Usage
-----
    python scripts/export_hrnet_onnx.py kp        # exports models/hrnet_kp.onnx
    python scripts/export_hrnet_onnx.py lines     # exports models/hrnet_lines.onnx
    python scripts/export_hrnet_onnx.py both      # both

Source weights are auto-downloaded from
``https://github.com/mguti97/PnLCalib/releases/tag/v1.0.0`` and cached under
``models/SV_kp.pt`` / ``models/SV_lines.pt`` if not present.

Exported ONNX has fixed input shape ``(1, 3, 540, 960)`` (PnLCalib's native
inference resolution). Opset 17 (good ONNX Runtime support).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(PROJECT_ROOT))

from app.vendor.pnlcalib.model.cls_hrnet import get_cls_net  # noqa: E402
from app.vendor.pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PNLCALIB_BASE = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"

CFG_KP = PROJECT_ROOT / "app" / "vendor" / "pnlcalib" / "config" / "hrnetv2_w48.yaml"
CFG_LINES = PROJECT_ROOT / "app" / "vendor" / "pnlcalib" / "config" / "hrnetv2_w48_l.yaml"

INPUT_SHAPE = (1, 3, 540, 960)  # NCHW


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        logger.info(f"already present: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading {url} -> {dst} (~258MB)")
    with urllib.request.urlopen(url) as response, dst.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024 * 4
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                logger.info(f"  ... {downloaded / 1e6:6.1f}/{total / 1e6:6.1f} MB ({pct:5.1f}%)")


def _load_kp_model() -> torch.nn.Module:
    cfg = yaml.safe_load(CFG_KP.read_text())
    model = get_cls_net(cfg)
    return model


def _load_lines_model() -> torch.nn.Module:
    cfg = yaml.safe_load(CFG_LINES.read_text())
    model = get_cls_net_l(cfg)
    return model


def export_kp() -> Path:
    weights_path = MODELS_DIR / "SV_kp.pt"
    onnx_path = MODELS_DIR / "hrnet_kp.onnx"
    _download(f"{PNLCALIB_BASE}/SV_kp", weights_path)

    model = _load_kp_model()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    logger.info(f"exporting -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    sz = onnx_path.stat().st_size / 1e6
    logger.info(f"OK: {onnx_path} ({sz:.1f} MB)")
    return onnx_path


def export_lines() -> Path:
    weights_path = MODELS_DIR / "SV_lines.pt"
    onnx_path = MODELS_DIR / "hrnet_lines.onnx"
    _download(f"{PNLCALIB_BASE}/SV_lines", weights_path)

    model = _load_lines_model()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    logger.info(f"exporting -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    sz = onnx_path.stat().st_size / 1e6
    logger.info(f"OK: {onnx_path} ({sz:.1f} MB)")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=["kp", "lines", "both"], help="model(s) to export")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.which in ("kp", "both"):
        export_kp()
    if args.which in ("lines", "both"):
        export_lines()


if __name__ == "__main__":
    main()
