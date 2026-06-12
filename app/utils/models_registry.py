"""Registry of all ML model artifacts used by the GSR pipeline.

Adds support for multiple models with different download backends
(Google Drive via gdown, direct URL via requests/urllib).
The registry is the single source of truth for paths so services
just ask for a model by logical name.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a downloadable model artifact.

    Resolution priority for ensure_model:
        1. env override `MODELS__{NAME_UPPER}__PATH`
        2. file already at target path (in models_dir)
        3. local_source — file copied from somewhere on disk
        4. gdrive_id — downloaded via gdown
        5. url — downloaded via urllib
    """

    name: str
    filename: str
    gdrive_id: Optional[str] = None
    url: Optional[str] = None
    local_source: Optional[Path] = None
    extra_files: tuple[tuple[str, Optional[str], Optional[str], Optional[Path]], ...] = field(
        default_factory=tuple
    )
    sha256: Optional[str] = None
    description: str = ""

    def local_path(self, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
        return models_dir / self.filename


REGISTRY: Dict[str, ModelSpec] = {
    "yolo_detector": ModelSpec(
        name="yolo_detector",
        filename="best.onnx",
        gdrive_id="1pkRFUd-YuXMjjMNcrH_lkGLM_HKKeQaG",
        sha256="e720c20237194f1c1bfd8775726adfa5641dfe08e1af73128088b173c29f93ef",
        description="YOLOv5lu detector (player/goalkeeper/referee/ball).",
    ),
    "deimv2_detector": ModelSpec(
        name="deimv2_detector",
        filename="deimv2_m_896.onnx",
        gdrive_id="1agddoWAy2CZdfh8cFGBVKMIG6MbS5Wwk",
        sha256="c0f7ced7ff783f4ae318f4728eb567a97e2c2e55c744ba0bb3b95ed6a0052934",
        description=(
            "DEIMv2-DINOv3 M @ 896 detector (DETR-style, postproc in-graph). "
            "Inputs: images [N,3,896,896] f32, orig_target_sizes [N,2] i64. "
            "Outputs: labels [N,300] i64, boxes [N,300,4] xyxy f32 (input coords), "
            "scores [N,300] f32. 5 classes incl background@0; service cls_id = "
            "label - 1. Re-export from best_stg2.pth via DEIMv2 export_l_model.py."
        ),
    ),
    "visibility_gate": ModelSpec(
        name="visibility_gate",
        filename="visibility_gate.onnx",
        gdrive_id="1k-RbqYYUHWoKyS3HdSyvEjsLCmMk-1qE",
        sha256="ab18a9be0c3990e930ac5ab87c5a0fabaabddefba3eb77ffb110f6db5c68f0a2",
        description=(
            "Binary classifier: is jersey number visible on a player crop. "
            "Source: jersey-visibility-project, ShuffleNetV2."
        ),
    ),
    "jersey_ocr": ModelSpec(
        name="jersey_ocr",
        filename="jersey_ocr.onnx",
        gdrive_id="1bfhGV1L0__W8w5wDV0J7s8ouEXFXGO5H",
        sha256="f1f8cb1b57b1eaf00ce61c63d6c93513d3210e3f3aaab322e03b97e264aa2100",
        extra_files=(
            (
                "jersey_ocr.onnx.data",
                "1DgHbIJ5r-A5Owsqg2oNMULuUmrmSyAwd",
                None,
                None,
            ),
        ),
        description=(
            "Two-head jersey number OCR (logits_tens, logits_units). "
            "Source: jersey-ocr-project, ConvNeXt-Tiny. External weights in "
            "jersey_ocr.onnx.data (must sit next to the .onnx)."
        ),
    ),
    "hrnet_kp": ModelSpec(
        name="hrnet_kp",
        filename="hrnet_kp.onnx",
        gdrive_id="1aV-86uvJ-WZ-OQkgzYzqvSwuRDB-KTJ3",
        sha256="9e55024e0ae73561ef29b97c456bf812c4fc2d68cdaef52390ed40157c6d6f37",
        description=(
            "HRNet keypoints detector for soccer field calibration. "
            "Re-export via `python scripts/export_hrnet_onnx.py kp` from "
            "PnLCalib SV_kp weights if needed."
        ),
    ),
    "hrnet_lines": ModelSpec(
        name="hrnet_lines",
        filename="hrnet_lines.onnx",
        gdrive_id="1R3euywxJqXodFGyHIiWpJU6Y118ykSTR",
        sha256="c71d1e28e5a7182c49b35bdb363be0328def46dee6ef79bd9915ca9ed9a03946",
        description=(
            "HRNet line extremities detector for soccer field calibration. "
            "Re-export via `python scripts/export_hrnet_onnx.py lines` from "
            "PnLCalib SV_lines weights if needed."
        ),
    ),
    "dinov3_embedder": ModelSpec(
        name="dinov3_embedder",
        filename="dinov3_vits16plus.onnx",
        gdrive_id="1eJNBUo672fvriNHaTcMkSbkS0e20jqkx",
        sha256="0f5c865f612cb166180da5f5681ce91bb9757a6617ecf641b5f78bbf89c8c729",
        description=(
            "DINOv3 ViT-S+/16 embedder (28.7M params, 384-d pooler_output). "
            "Used as a generic appearance feature for both BoT-SORT ReID and "
            "team clustering. Run `python scripts/export_dinov3_onnx.py` to "
            "(re)export from facebook/dinov3-vits16plus-pretrain-lvd1689m. "
            "License: DINOv3 (Meta), see https://ai.meta.com/resources/"
            "models-and-libraries/dinov3-license/."
        ),
    ),
}


def _verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _download_gdrive(file_id: str, output_path: Path, quiet: bool = False) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download from Google Drive. pip install gdown"
        ) from exc
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(output_path), quiet=quiet)


def _download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _copy_local(src: Path, dst: Path) -> None:
    import shutil

    if not src.exists():
        raise FileNotFoundError(f"local_source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Copying local source {src} -> {dst}")
    shutil.copy2(src, dst)


def _ensure_extra_files(
    spec: ModelSpec,
    target: Path,
    *,
    auto_download: bool = True,
    quiet: bool = False,
) -> None:
    """Materialize spec.extra_files next to ``target`` (e.g. ONNX external
    weights). Must run even when the main file is already on disk."""
    for extra_filename, extra_gdrive, extra_url, extra_local in spec.extra_files:
        extra_target = target.parent / extra_filename
        if extra_target.exists():
            continue
        if not auto_download:
            raise FileNotFoundError(
                f"Extra file {extra_target} for model '{spec.name}' is missing "
                "and auto_download=False"
            )
        if extra_local and extra_local.exists():
            _copy_local(extra_local, extra_target)
        elif extra_gdrive:
            _download_gdrive(extra_gdrive, extra_target, quiet=quiet)
        elif extra_url:
            _download_url(extra_url, extra_target)
        else:
            logger.warning(
                f"Extra file {extra_filename} for model '{spec.name}' is not "
                "configured; expecting it to be present locally if needed."
            )


def ensure_model(
    name: str,
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    auto_download: bool = True,
    quiet: bool = False,
) -> Path:
    """Resolve a model by registry name, downloading if missing.

    Honors `MODELS__{NAME}__PATH` env override (uppercased name) for ad-hoc paths.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")

    spec = REGISTRY[name]

    env_key = f"MODELS__{name.upper()}__PATH"
    if env_key in os.environ:
        return Path(os.environ[env_key])

    target = spec.local_path(models_dir)

    if target.exists():
        if spec.sha256 and not _verify_sha256(target, spec.sha256):
            logger.warning(f"sha256 mismatch for {target}, re-downloading")
            target.unlink(missing_ok=True)
        else:
            # main file is fine, but external weights may still be missing
            _ensure_extra_files(spec, target, auto_download=auto_download, quiet=quiet)
            return target

    if not auto_download:
        raise FileNotFoundError(
            f"Model '{name}' not found at {target} and auto_download=False"
        )

    if spec.local_source and spec.local_source.exists():
        _copy_local(spec.local_source, target)
    elif spec.gdrive_id:
        _download_gdrive(spec.gdrive_id, target, quiet=quiet)
    elif spec.url:
        _download_url(spec.url, target)
    else:
        raise FileNotFoundError(
            f"Model '{name}' has no source configured and is not on disk at {target}. "
            f"{spec.description}"
        )

    _ensure_extra_files(spec, target, auto_download=auto_download, quiet=quiet)

    if not target.exists():
        raise RuntimeError(f"Failed to materialize model '{name}' at {target}")

    if spec.sha256 and not _verify_sha256(target, spec.sha256):
        raise RuntimeError(f"sha256 verification failed for {target}")

    return target


def ensure_models(
    names: Iterable[str],
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    auto_download: bool = True,
    skip_missing_remotes: bool = True,
) -> Dict[str, Optional[Path]]:
    """Best-effort batch ensure. Returns {name: path or None on failure}.

    `skip_missing_remotes=True` lets startup proceed even if some models
    are not yet uploaded; services that need them will fail loudly later.
    """
    out: Dict[str, Optional[Path]] = {}
    for name in names:
        try:
            out[name] = ensure_model(
                name, models_dir=models_dir, auto_download=auto_download
            )
        except Exception as exc:
            if skip_missing_remotes:
                logger.warning(f"Could not ensure model '{name}': {exc}")
                out[name] = None
            else:
                raise
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ensure GSR model artifacts are present locally")
    parser.add_argument("--name", type=str, default=None, help="Single model to ensure (omit for all)")
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
        help=f"Where to store models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on missing remote sources (default: skip and warn)",
    )
    args = parser.parse_args()

    targets = [args.name] if args.name else list(REGISTRY)
    results = ensure_models(
        targets,
        models_dir=args.models_dir,
        skip_missing_remotes=not args.strict,
    )
    for name, path in results.items():
        marker = "OK " if path and path.exists() else "MISS"
        print(f"{marker} {name:20s} -> {path or REGISTRY[name].local_path(args.models_dir)}")
