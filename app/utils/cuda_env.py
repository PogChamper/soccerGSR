"""WSL2-friendly CUDA bootstrap for onnxruntime-gpu.

Without this, `onnxruntime-gpu >= 1.19` on WSL2 crashes with
``CUDA failure 100: no CUDA-capable device`` because libcuda.so.1 from
``/usr/lib/wsl/lib`` is not in LD_LIBRARY_PATH and ORT's preloader
needs to be invoked explicitly.

Call ``bootstrap()`` BEFORE any ``onnxruntime.InferenceSession(...)``.
Idempotent.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

WSL_LIB = "/usr/lib/wsl/lib"
_BOOTSTRAPPED = False


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
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
    logger.info(f"cuda_env: prepended {WSL_LIB} to LD_LIBRARY_PATH")


def bootstrap(*, preload_dlls: bool = True, force: bool = False) -> bool:
    """Prepare environment so onnxruntime-gpu can load CUDA on WSL2.

    Returns True if at least one CUDA execution provider is registered
    after preload, False otherwise. Safe to call multiple times.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not force:
        return True

    if _is_wsl():
        _ensure_ld_path()
    else:
        logger.debug("cuda_env: not WSL, skipping LD path adjustment")

    if not preload_dlls:
        _BOOTSTRAPPED = True
        return False

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("cuda_env: onnxruntime is not installed")
        return False

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
            logger.info("cuda_env: ort.preload_dlls(cuda=True, cudnn=True) ok")
        except Exception as exc:
            logger.warning(f"cuda_env: preload_dlls failed: {exc}")
    else:
        logger.debug(
            "cuda_env: ort.preload_dlls not available "
            "(onnxruntime < 1.19?), skipping"
        )

    providers = ort.get_available_providers()
    has_cuda = any(p.startswith("CUDA") for p in providers)
    logger.info(f"cuda_env: available providers = {providers}")
    _BOOTSTRAPPED = True
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
    """Same as ``get_providers`` but each provider gets explicit options.

    On WSL2 + CUDA13 + cudnn 9.1 + ORT 1.20, the default cudnn frontend path
    can fail with ``CUDNN_BACKEND_API_FAILED`` for big convs (HRNet on
    540x960). Forcing HEURISTIC search and capping workspace forces the legacy
    cudnn algorithm selection path which is stable.
    """
    bootstrap()
    import onnxruntime as ort

    out = []
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": int(gpu_mem_limit_gb * 1024 ** 3),
            "cudnn_conv_algo_search": cudnn_conv_algo_search,
            "do_copy_in_default_stream": True,
            "cudnn_conv_use_max_workspace": "0",
        }
        out.append(("CUDAExecutionProvider", cuda_opts))
    out.append("CPUExecutionProvider")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ok = bootstrap()
    print(f"CUDA available: {ok}")
    print(f"Providers: {get_providers()}")
