"""WSL2-friendly CUDA bootstrap for ONNX Runtime 1.21 or newer.

Without this, ONNX Runtime on WSL2 can fail with
``CUDA failure 100: no CUDA-capable device`` because libcuda.so.1 from
``/usr/lib/wsl/lib`` is not in LD_LIBRARY_PATH and ORT's preloader
needs to be invoked explicitly.

Call ``bootstrap()`` BEFORE any ``onnxruntime.InferenceSession(...)``.
Idempotent.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

WSL_LIB = "/usr/lib/wsl/lib"
_CUDA_AVAILABLE: bool | None = None
_WSL_DRIVER: ctypes.CDLL | None = None


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _validate_ort_installation() -> str | None:
    cpu_version = _distribution_version("onnxruntime")
    gpu_version = _distribution_version("onnxruntime-gpu")
    if cpu_version and gpu_version:
        raise RuntimeError(
            "onnxruntime and onnxruntime-gpu are installed together and share one "
            "Python namespace. Uninstall both packages, then install only "
            "onnxruntime-gpu."
        )
    return gpu_version


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _ensure_ld_path() -> None:
    if not Path(WSL_LIB).is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = current.split(":") if current else []
    if WSL_LIB in parts:
        return
    parts.insert(0, WSL_LIB)
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)
    logger.info("cuda_env: prepended %s to LD_LIBRARY_PATH", WSL_LIB)


def _preload_wsl_driver() -> None:
    global _WSL_DRIVER
    if _WSL_DRIVER is not None:
        return
    driver_path = Path(WSL_LIB) / "libcuda.so.1"
    if not driver_path.is_file():
        logger.warning("cuda_env: WSL CUDA driver not found at %s", driver_path)
        return
    try:
        _WSL_DRIVER = ctypes.CDLL(str(driver_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        logger.warning("cuda_env: could not preload %s: %s", driver_path, exc)
    else:
        logger.info("cuda_env: preloaded %s", driver_path)


def bootstrap(*, preload_dlls: bool = True) -> bool:
    """Prepare environment so onnxruntime-gpu can load CUDA on WSL2.

    Returns True if at least one CUDA execution provider is registered
    after preload, False otherwise. Safe to call multiple times.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE

    gpu_version = _validate_ort_installation()

    if _is_wsl():
        _ensure_ld_path()
        _preload_wsl_driver()
    else:
        logger.debug("cuda_env: not WSL, skipping LD path adjustment")

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("cuda_env: onnxruntime is not installed")
        _CUDA_AVAILABLE = False
        return False

    if preload_dlls and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True, directory="")
            logger.info("cuda_env: requested CUDA and cuDNN preload from site-packages")
        except Exception as exc:
            logger.warning("cuda_env: preload_dlls failed: %s", exc)
    elif preload_dlls:
        logger.warning("cuda_env: ort.preload_dlls requires onnxruntime-gpu >= 1.21")

    providers = ort.get_available_providers()
    has_cuda = any(p.startswith("CUDA") for p in providers)
    logger.info("cuda_env: available providers=%s", providers)
    if gpu_version and not has_cuda:
        logger.warning(
            "cuda_env: onnxruntime-gpu %s is installed but CUDAExecutionProvider is unavailable",
            gpu_version,
        )
    _CUDA_AVAILABLE = has_cuda
    return has_cuda


def get_providers(prefer_gpu: bool = True) -> list[str]:
    """Return ORT providers list with sane defaults for our setup."""
    bootstrap()
    import onnxruntime as ort

    providers: list[str] = []
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def get_providers_with_options(
    prefer_gpu: bool = True,
    *,
    gpu_mem_limit_gb: float = 6.0,
    cudnn_conv_algo_search: str = "HEURISTIC",
) -> list:
    """Return CUDA/CPU providers with a bounded arena and heuristic conv search."""
    bootstrap()
    import onnxruntime as ort

    out = []
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": int(gpu_mem_limit_gb * 1024**3),
            "cudnn_conv_algo_search": cudnn_conv_algo_search,
            "do_copy_in_default_stream": True,
            "cudnn_conv_use_max_workspace": "0",
        }
        out.append(("CUDAExecutionProvider", cuda_opts))
    out.append("CPUExecutionProvider")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        ok = bootstrap()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    print(f"CUDA available: {ok}")
    print(f"Providers: {get_providers()}")
