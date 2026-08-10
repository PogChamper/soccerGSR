from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.utils import models_registry
from app.utils.models_registry import ExtraArtifact, ModelSpec


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _register(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    payload: bytes,
    url: str | None = "https://models.invalid/model.onnx",
    extra_files: tuple[ExtraArtifact, ...] = (),
) -> ModelSpec:
    spec = ModelSpec(
        name=name,
        filename=f"{name}.onnx",
        url=url,
        sha256=_sha256(payload),
        extra_files=extra_files,
    )
    monkeypatch.setitem(models_registry.REGISTRY, name, spec)
    monkeypatch.delenv(f"MODELS__{name.upper()}__PATH", raising=False)
    return spec


def test_override_requires_a_file_and_matching_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"verified model"
    spec = _register(monkeypatch, name="override_test", payload=payload)
    env_key = "MODELS__OVERRIDE_TEST__PATH"

    directory = tmp_path / "directory"
    directory.mkdir()
    monkeypatch.setenv(env_key, str(directory))
    with pytest.raises(FileNotFoundError, match="does not point to a file"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path / "models")

    override = tmp_path / "override.onnx"
    override.write_bytes(b"corrupt")
    monkeypatch.setenv(env_key, str(override))
    with pytest.raises(RuntimeError, match="sha256 verification failed"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path / "models")

    override.write_bytes(payload)
    assert models_registry.ensure_model(spec.name, models_dir=tmp_path / "models") == override


def test_override_checks_external_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"verified model"
    spec = _register(
        monkeypatch,
        name="external_test",
        payload=payload,
        extra_files=(ExtraArtifact("weights.data"),),
    )
    override = tmp_path / "override.onnx"
    override.write_bytes(payload)
    monkeypatch.setenv("MODELS__EXTERNAL_TEST__PATH", str(override))

    with pytest.raises(FileNotFoundError, match="weights.data"):
        models_registry.ensure_model(
            spec.name,
            models_dir=tmp_path / "models",
            auto_download=False,
        )


def test_corrupt_external_artifact_is_replaced_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_payload = b"verified model"
    external_payload = b"verified external weights"
    spec = _register(
        monkeypatch,
        name="external_checksum_test",
        payload=model_payload,
        extra_files=(
            ExtraArtifact(
                "weights.data",
                url="https://models.invalid/weights.data",
                sha256=_sha256(external_payload),
            ),
        ),
    )
    target = spec.local_path(tmp_path)
    target.write_bytes(model_payload)
    external_target = target.with_name("weights.data")
    corrupt = b"existing corrupt weights"
    external_target.write_bytes(corrupt)

    def download(_url: str, temporary: Path) -> None:
        assert external_target.read_bytes() == corrupt
        temporary.write_bytes(external_payload)

    monkeypatch.setattr(models_registry, "_download_url", download)

    assert models_registry.ensure_model(spec.name, models_dir=tmp_path) == target
    assert external_target.read_bytes() == external_payload
    assert list(tmp_path.glob(f".{external_target.name}.*.part")) == []


def test_corrupt_external_artifact_is_preserved_when_download_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_payload = b"verified model"
    external_payload = b"verified external weights"
    spec = _register(
        monkeypatch,
        name="external_preserve_test",
        payload=model_payload,
        extra_files=(
            ExtraArtifact(
                "weights.data",
                url="https://models.invalid/weights.data",
                sha256=_sha256(external_payload),
            ),
        ),
    )
    target = spec.local_path(tmp_path)
    target.write_bytes(model_payload)
    external_target = target.with_name("weights.data")
    corrupt = b"existing corrupt weights"
    external_target.write_bytes(corrupt)

    with pytest.raises(RuntimeError, match="sha256 verification failed"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path, auto_download=False)

    assert external_target.read_bytes() == corrupt


def test_corrupt_artifact_is_preserved_when_download_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _register(monkeypatch, name="preserve_test", payload=b"verified model")
    target = spec.local_path(tmp_path)
    corrupt = b"do not delete"
    target.write_bytes(corrupt)

    with pytest.raises(RuntimeError, match="sha256 verification failed"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path, auto_download=False)

    assert target.read_bytes() == corrupt


def test_download_replaces_target_only_after_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"verified model"
    spec = _register(monkeypatch, name="atomic_test", payload=payload)
    target = spec.local_path(tmp_path)
    corrupt = b"existing corrupt model"
    target.write_bytes(corrupt)

    def download(_url: str, temporary: Path) -> None:
        assert target.read_bytes() == corrupt
        assert temporary.parent == target.parent
        assert temporary.name.startswith(f".{target.name}.")
        temporary.write_bytes(payload)

    monkeypatch.setattr(models_registry, "_download_url", download)

    assert models_registry.ensure_model(spec.name, models_dir=tmp_path) == target
    assert target.read_bytes() == payload
    assert list(tmp_path.glob(f".{target.name}.*.part")) == []


def test_failed_download_preserves_target_and_cleans_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _register(monkeypatch, name="failure_test", payload=b"verified model")
    target = spec.local_path(tmp_path)
    corrupt = b"existing corrupt model"
    target.write_bytes(corrupt)

    def download(_url: str, temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise ConnectionError("download interrupted")

    monkeypatch.setattr(models_registry, "_download_url", download)

    with pytest.raises(ConnectionError, match="download interrupted"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path)

    assert target.read_bytes() == corrupt
    assert list(tmp_path.glob(f".{target.name}.*.part")) == []


def test_failed_download_verification_preserves_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _register(monkeypatch, name="checksum_test", payload=b"verified model")
    target = spec.local_path(tmp_path)
    corrupt = b"existing corrupt model"
    target.write_bytes(corrupt)

    monkeypatch.setattr(
        models_registry,
        "_download_url",
        lambda _url, temporary: temporary.write_bytes(b"wrong download"),
    )

    with pytest.raises(RuntimeError, match="sha256 verification failed for downloaded"):
        models_registry.ensure_model(spec.name, models_dir=tmp_path)

    assert target.read_bytes() == corrupt
    assert list(tmp_path.glob(f".{target.name}.*.part")) == []


def test_osnet_without_source_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MODELS__OSNET_REID__PATH", raising=False)

    with pytest.raises(FileNotFoundError, match="has no download source") as exc_info:
        models_registry.ensure_model("osnet_reid", models_dir=tmp_path)

    message = str(exc_info.value)
    assert str(tmp_path / "osnet_x1_0_soccernet.onnx") in message
    assert "MODELS__OSNET_REID__PATH" in message


def test_corrupt_osnet_without_source_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("MODELS__OSNET_REID__PATH", raising=False)
    target = tmp_path / "osnet_x1_0_soccernet.onnx"
    corrupt = b"corrupt osnet"
    target.write_bytes(corrupt)

    with pytest.raises(RuntimeError, match="has no download source"):
        models_registry.ensure_model("osnet_reid", models_dir=tmp_path)

    assert target.read_bytes() == corrupt


def test_bulk_resolution_skips_only_unconfigured_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unconfigured = _register(
        monkeypatch,
        name="unconfigured_test",
        payload=b"verified model",
        url=None,
    )
    result = models_registry.ensure_models([unconfigured.name], models_dir=tmp_path)
    assert result == {unconfigured.name: None}

    corrupt = unconfigured.local_path(tmp_path)
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="sha256 verification failed"):
        models_registry.ensure_models([unconfigured.name], models_dir=tmp_path)

    with pytest.raises(KeyError, match="Unknown model"):
        models_registry.ensure_models(["unknown"], models_dir=tmp_path)


def test_bulk_resolution_handles_unconfigured_external_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"verified model"
    spec = _register(
        monkeypatch,
        name="unconfigured_external_test",
        payload=payload,
        extra_files=(ExtraArtifact("weights.data"),),
    )
    spec.local_path(tmp_path).write_bytes(payload)

    assert models_registry.ensure_models(
        [spec.name],
        models_dir=tmp_path,
        auto_download=False,
    ) == {spec.name: None}

    with pytest.raises(FileNotFoundError, match="has no download source"):
        models_registry.ensure_models(
            [spec.name],
            models_dir=tmp_path,
            auto_download=False,
            skip_unconfigured=False,
        )
