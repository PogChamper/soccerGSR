"""Export DINOv3 ViT-S+/16 to ONNX.

Output: pooler_output of shape (B, 384) — used as a generic appearance
embedding for both BoT-SORT ReID and team clustering.

Usage:
    python scripts/export_dinov3_onnx.py
    python scripts/export_dinov3_onnx.py --model facebook/dinov3-vits16plus-pretrain-lvd1689m

Notes
-----
* The DINOv3 weights are gated on Hugging Face. Make sure your HF token is
  authenticated and that you have accepted the DINOv3 license at
  https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m before
  running this script.
* Dynamic batch axis is exported (axis 0). Spatial size is fixed at 224x224
  (default DINOv3 image processor size). Bigger inputs are possible but
  must be multiples of patch_size=16; we keep the export rigid because all
  player crops will be normalised to 224x224 anyway.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.utils.models_registry import DEFAULT_MODELS_DIR

DEFAULT_MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
DEFAULT_OUT = DEFAULT_MODELS_DIR / "dinov3_vits16plus.onnx"


class _PoolerOnly(torch.nn.Module):
    """Wraps DINOv3 to expose only the pooler_output tensor (cleaner ONNX graph)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values=pixel_values)
        return out.pooler_output


def export(model_id: str, out_path: Path, opset: int = 17) -> Path:
    from transformers import AutoModel

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} ...")
    base = AutoModel.from_pretrained(model_id).eval()
    model = _PoolerOnly(base).eval()

    dummy = torch.randn(1, 3, 224, 224)
    print(f"exporting -> {out_path}")
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  saved {out_path} ({size_mb:.1f} MB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id")
    p.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output .onnx path")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export(args.model, args.out, opset=args.opset)


if __name__ == "__main__":
    main()
