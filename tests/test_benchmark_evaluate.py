from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.evaluate import discover_sequences, resolve_sequences, validate_predictions


def _add_labels(gt_root: Path, split: str, sequence: str) -> None:
    sequence_root = gt_root / split / sequence
    sequence_root.mkdir(parents=True)
    (sequence_root / "Labels-GameState.json").write_text("{}", encoding="utf-8")


def _add_prediction(trackers_root: Path, split: str, tag: str, sequence: str) -> None:
    data_root = trackers_root / f"SoccerNetGS-{split}" / tag / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / f"{sequence}.json").write_text("{}", encoding="utf-8")


def test_discover_sequences_uses_only_labelled_gt_directories(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    trackers_root = tmp_path / "trackers"
    _add_labels(gt_root, "test", "SNGS-002")
    _add_labels(gt_root, "test", "SNGS-001")
    (gt_root / "test" / "SNGS-unlabelled").mkdir()
    _add_prediction(trackers_root, "test", "ours", "SNGS-stale")

    assert discover_sequences(gt_root, "test") == ("SNGS-001", "SNGS-002")
    assert resolve_sequences(gt_root, "test", "all") == ("SNGS-001", "SNGS-002")


def test_explicit_sequence_selection_is_trimmed_and_keeps_order(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    _add_labels(gt_root, "valid", "SNGS-021")
    _add_labels(gt_root, "valid", "SNGS-022")

    assert resolve_sequences(gt_root, "valid", " SNGS-022, SNGS-021 ") == (
        "SNGS-022",
        "SNGS-021",
    )


def test_explicit_sequence_requires_ground_truth_labels(tmp_path: Path) -> None:
    gt_root = tmp_path / "gt"
    (gt_root / "test").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="SNGS-404/Labels-GameState.json"):
        resolve_sequences(gt_root, "test", "SNGS-404")


def test_validation_fails_on_first_missing_selected_prediction(tmp_path: Path) -> None:
    trackers_root = tmp_path / "trackers"
    _add_prediction(trackers_root, "test", "ours", "SNGS-001")

    expected = trackers_root / "SoccerNetGS-test" / "ours" / "data" / "SNGS-002.json"
    with pytest.raises(FileNotFoundError, match=str(expected)):
        validate_predictions(trackers_root, "test", "ours", ("SNGS-001", "SNGS-002"))
