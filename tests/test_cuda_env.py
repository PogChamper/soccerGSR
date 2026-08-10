import sys
from types import SimpleNamespace

import pytest

from app.utils import cuda_env


def test_bootstrap_returns_cached_cpu_state(monkeypatch) -> None:
    monkeypatch.setattr(cuda_env, "_CUDA_AVAILABLE", False)
    monkeypatch.setattr(
        cuda_env,
        "_is_wsl",
        lambda: (_ for _ in ()).throw(AssertionError("cache was ignored")),
    )

    assert cuda_env.bootstrap() is False


def test_bootstrap_rejects_cpu_and_gpu_wheels_together(monkeypatch) -> None:
    versions = {"onnxruntime": "1.20.1", "onnxruntime-gpu": "1.21.0"}
    monkeypatch.setattr(cuda_env, "_CUDA_AVAILABLE", None)
    monkeypatch.setattr(cuda_env, "_distribution_version", versions.get)

    with pytest.raises(RuntimeError, match="installed together"):
        cuda_env.bootstrap()


def test_bootstrap_preloads_nvidia_site_packages(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    fake_ort = SimpleNamespace(
        preload_dlls=lambda **kwargs: calls.append(kwargs),
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(cuda_env, "_CUDA_AVAILABLE", None)
    monkeypatch.setattr(
        cuda_env, "_distribution_version", lambda name: "1.21.1" if name.endswith("-gpu") else None
    )
    monkeypatch.setattr(cuda_env, "_is_wsl", lambda: False)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert cuda_env.bootstrap() is True
    assert calls == [{"cuda": True, "cudnn": True, "directory": ""}]


def test_bootstrap_without_preload_still_detects_cuda(monkeypatch) -> None:
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(cuda_env, "_CUDA_AVAILABLE", None)
    monkeypatch.setattr(
        cuda_env,
        "_distribution_version",
        lambda name: "1.21.1" if name.endswith("-gpu") else None,
    )
    monkeypatch.setattr(cuda_env, "_is_wsl", lambda: False)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    assert cuda_env.bootstrap(preload_dlls=False) is True


def test_preload_wsl_driver_once(monkeypatch, tmp_path) -> None:
    driver_path = tmp_path / "libcuda.so.1"
    driver_path.touch()
    handle = object()
    calls: list[tuple[str, int]] = []

    def load_driver(path: str, *, mode: int):
        calls.append((path, mode))
        return handle

    monkeypatch.setattr(cuda_env, "WSL_LIB", str(tmp_path))
    monkeypatch.setattr(cuda_env, "_WSL_DRIVER", None)
    monkeypatch.setattr(cuda_env.ctypes, "CDLL", load_driver)

    cuda_env._preload_wsl_driver()
    cuda_env._preload_wsl_driver()

    assert calls == [(str(driver_path), cuda_env.ctypes.RTLD_GLOBAL)]
    assert cuda_env._WSL_DRIVER is handle
