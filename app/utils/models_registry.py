"""Download and verify runtime model artifacts."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ExtraArtifact:
    filename: str
    gdrive_id: str | None = None
    url: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Describe one runtime model artifact.

    Resolution priority for ensure_model:
        1. env override `MODELS__{NAME_UPPER}__PATH`
        2. file already at target path (in models_dir)
        3. gdrive_id - downloaded via gdown
        4. url - downloaded via urllib
    """

    name: str
    filename: str
    gdrive_id: str | None = None
    url: str | None = None
    extra_files: tuple[ExtraArtifact, ...] = ()
    sha256: str | None = None
    description: str = ""

    def local_path(self, models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
        return models_dir / self.filename


REGISTRY: dict[str, ModelSpec] = {
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
            "label - 1."
        ),
    ),
    "visibility_gate": ModelSpec(
        name="visibility_gate",
        filename="visibility_gate.onnx",
        gdrive_id="1k-RbqYYUHWoKyS3HdSyvEjsLCmMk-1qE",
        sha256="ab18a9be0c3990e930ac5ab87c5a0fabaabddefba3eb77ffb110f6db5c68f0a2",
        description=(
            "ShuffleNetV2 binary classifier for jersey-number visibility on a player crop."
        ),
    ),
    "jersey_ocr": ModelSpec(
        name="jersey_ocr",
        filename="jersey_ocr.onnx",
        gdrive_id="1bfhGV1L0__W8w5wDV0J7s8ouEXFXGO5H",
        sha256="f1f8cb1b57b1eaf00ce61c63d6c93513d3210e3f3aaab322e03b97e264aa2100",
        extra_files=(
            ExtraArtifact(
                filename="jersey_ocr.onnx.data",
                gdrive_id="1DgHbIJ5r-A5Owsqg2oNMULuUmrmSyAwd",
                sha256="ad040303f30c389d3a2c2b12be912ce90f41abe2b918a1d9b4e4050aed25a85f",
            ),
        ),
        description=(
            "ConvNeXt-Tiny two-head jersey-number OCR (logits_tens, logits_units). "
            "External weights in "
            "jersey_ocr.onnx.data (must sit next to the .onnx)."
        ),
    ),
    "hrnet_kp": ModelSpec(
        name="hrnet_kp",
        filename="hrnet_kp.onnx",
        gdrive_id="1aV-86uvJ-WZ-OQkgzYzqvSwuRDB-KTJ3",
        sha256="9e55024e0ae73561ef29b97c456bf812c4fc2d68cdaef52390ed40157c6d6f37",
        description="PnLCalib HRNet keypoint detector for field calibration.",
    ),
    "hrnet_lines": ModelSpec(
        name="hrnet_lines",
        filename="hrnet_lines.onnx",
        gdrive_id="1R3euywxJqXodFGyHIiWpJU6Y118ykSTR",
        sha256="c71d1e28e5a7182c49b35bdb363be0328def46dee6ef79bd9915ca9ed9a03946",
        description="PnLCalib HRNet line-extremity detector for field calibration.",
    ),
    "osnet_reid": ModelSpec(
        name="osnet_reid",
        filename="osnet_x1_0_soccernet.onnx",
        sha256="6d7a70bb28c309d91f970dbff190755a95bfed78089aa6a9831b1824914cb078",
        description=(
            "SoccerNet OSNet-x1.0 feature extractor. Input: float32 "
            "[N,3,256,128], RGB with ImageNet normalization. Output: "
            "float32 [N,512], normalized by the runtime."
        ),
    ),
}


def _verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _download_gdrive(file_id: str, output_path: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download from Google Drive; restore the locked "
            "runtime with `uv sync --locked --no-dev`"
        ) from exc
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gdown.download(url, str(output_path)) is None:
        raise RuntimeError(f"Failed to download Google Drive artifact {file_id}")


def _download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, output_path)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _download_atomic(
    target: Path,
    *,
    gdrive_id: str | None = None,
    url: str | None = None,
    sha256: str | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if gdrive_id:
            _download_gdrive(gdrive_id, temporary)
        elif url:
            _download_url(url, temporary)
        else:
            raise ValueError(f"No download source configured for {target.name}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded model is empty: {target.name}")
        if sha256 and not _verify_sha256(temporary, sha256):
            raise RuntimeError(f"sha256 verification failed for downloaded {target.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


class _SourceUnavailableError(FileNotFoundError):
    """A required artifact has no configured download source."""


def _ensure_extra_files(
    spec: ModelSpec,
    target: Path,
    *,
    auto_download: bool = True,
) -> None:
    """Ensure external model data is present next to the main artifact."""
    for artifact in spec.extra_files:
        extra_target = target.parent / artifact.filename
        if extra_target.is_file():
            if not artifact.sha256 or _verify_sha256(extra_target, artifact.sha256):
                continue
            if not auto_download:
                raise RuntimeError(f"sha256 verification failed for {extra_target}")
            if not (artifact.gdrive_id or artifact.url):
                raise RuntimeError(
                    f"sha256 verification failed for {extra_target}; no download source"
                )
            logger.warning("sha256 mismatch for %s; downloading a clean copy", extra_target)
        elif extra_target.exists():
            raise FileNotFoundError(f"Extra model artifact is not a file: {extra_target}")
        elif not (artifact.gdrive_id or artifact.url):
            raise _SourceUnavailableError(
                f"Extra file {extra_target} for model '{spec.name}' is missing and has no "
                "download source"
            )
        elif not auto_download:
            raise FileNotFoundError(
                f"Extra file {extra_target} for model '{spec.name}' is missing "
                "and auto_download=False"
            )
        _download_atomic(
            extra_target,
            gdrive_id=artifact.gdrive_id,
            url=artifact.url,
            sha256=artifact.sha256,
        )


def ensure_model(
    name: str,
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    auto_download: bool = True,
) -> Path:
    """Resolve a model by registry name, downloading if missing.

    Honors `MODELS__{NAME}__PATH` env override (uppercased name) for ad-hoc paths.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")

    spec = REGISTRY[name]

    env_key = f"MODELS__{name.upper()}__PATH"
    if env_key in os.environ:
        override = Path(os.environ[env_key]).expanduser()
        if not override.is_file():
            raise FileNotFoundError(f"{env_key} does not point to a file: {override}")
        if spec.sha256 and not _verify_sha256(override, spec.sha256):
            raise RuntimeError(f"sha256 verification failed for {override}")
        _ensure_extra_files(spec, override, auto_download=auto_download)
        return override

    target = spec.local_path(models_dir)
    has_source = bool(spec.gdrive_id or spec.url)

    if target.is_file():
        if spec.sha256 and not _verify_sha256(target, spec.sha256):
            if not auto_download:
                raise RuntimeError(f"sha256 verification failed for {target}")
            if not has_source:
                raise RuntimeError(
                    f"sha256 verification failed for {target}; model '{name}' has no "
                    f"download source. Replace the file or set {env_key}"
                )
            logger.warning("sha256 mismatch for %s; downloading a clean copy", target)
        else:
            _ensure_extra_files(spec, target, auto_download=auto_download)
            return target
    elif target.exists():
        raise FileNotFoundError(f"Model artifact is not a file: {target}")

    if not has_source:
        raise _SourceUnavailableError(
            f"Model '{name}' is not installed and has no download source. "
            f"Place the verified artifact at {target} or set {env_key}"
        )
    if not auto_download:
        raise FileNotFoundError(f"Model '{name}' not found at {target} and auto_download=False")

    _download_atomic(
        target,
        gdrive_id=spec.gdrive_id,
        url=spec.url,
        sha256=spec.sha256,
    )
    _ensure_extra_files(spec, target, auto_download=auto_download)
    return target


def ensure_models(
    names: Iterable[str],
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    auto_download: bool = True,
    skip_unconfigured: bool = True,
) -> dict[str, Path | None]:
    """Resolve several models, optionally skipping entries without a source.

    Integrity, configuration, and download errors are never suppressed.
    """
    out: dict[str, Path | None] = {}
    for name in names:
        try:
            out[name] = ensure_model(name, models_dir=models_dir, auto_download=auto_download)
        except _SourceUnavailableError as exc:
            if skip_unconfigured:
                logger.warning("Could not ensure model '%s': %s", name, exc)
                out[name] = None
            else:
                raise
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ensure GSR model artifacts are present locally")
    parser.add_argument(
        "--name", type=str, default=None, help="Single model to ensure (omit for all)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Where to store models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a registry entry has no download source",
    )
    args = parser.parse_args()

    targets = [args.name] if args.name else list(REGISTRY)
    results = ensure_models(
        targets,
        models_dir=args.models_dir,
        skip_unconfigured=not args.strict,
    )
    for name, path in results.items():
        marker = "OK " if path and path.is_file() else "MISS"
        print(f"{marker} {name:20s} -> {path or REGISTRY[name].local_path(args.models_dir)}")
