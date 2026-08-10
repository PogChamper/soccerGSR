from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmark.predictions import build_prediction_document, export_prediction


def _labels() -> dict:
    return {
        "images": [
            {"image_id": "1002", "file_name": "000002.jpg", "width": 100, "height": 50},
            {"image_id": "1001", "file_name": "000001.jpg", "width": 100, "height": 50},
        ],
        "categories": [
            {"id": 1, "name": "player", "supercategory": "object"},
            {"id": 2, "name": "goalkeeper", "supercategory": "object"},
            {"id": 3, "name": "referee", "supercategory": "object"},
            {"id": 4, "name": "ball", "supercategory": "object"},
            {"id": 5, "name": "pitch", "supercategory": "pitch"},
        ],
    }


def _state() -> dict:
    return {
        "meta": {"filename": "clip.mp4", "width": 100, "height": 50, "frame_count": 2},
        "frames": [
            {
                "frame_idx": 0,
                "width": 100,
                "height": 50,
                "H_img2world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            {
                "frame_idx": 1,
                "width": 100,
                "height": 50,
                "H_img2world": None,
            },
        ],
        "tracks": {
            "7": {
                "track_id": 7,
                "cls_id": 0,
                "cls_name": "player",
                "team_id": 0,
                "team_label": "team_left",
                "jersey_number": 10,
            },
            "8": {
                "track_id": 8,
                "cls_id": 1,
                "cls_name": "goalkeeper",
                "team_id": 1,
                "team_label": "goalkeeper_right",
                "jersey_number": 1,
            },
        },
        "observations": [
            {
                "frame_idx": 1,
                "track_id": 8,
                "cls_id": 1,
                "bbox_xyxy": [30, 10, 40, 45],
                "det_confidence": 0.7,
                "pitch_xy": [-40, 5],
                "synthetic": True,
            },
            {
                "frame_idx": 0,
                "track_id": 7,
                "cls_id": 0,
                "bbox_xyxy": [0, 0, 2, 4],
                "det_confidence": 0.8,
                "pitch_xy": [11, 22],
                "synthetic": False,
            },
            {
                "frame_idx": 0,
                "track_id": None,
                "cls_id": 0,
                "bbox_xyxy": [3, 0, 5, 4],
                "det_confidence": 0.6,
                "pitch_xy": None,
                "synthetic": False,
            },
        ],
    }


def test_builds_evaluator_schema_with_canonical_attributes() -> None:
    labels = _labels()

    output = build_prediction_document(_state(), labels)

    assert output["images"] == labels["images"]
    assert output["images"] is not labels["images"]
    assert [prediction["image_id"] for prediction in output["predictions"]] == ["1001", "1002"]

    player, goalkeeper = output["predictions"]
    assert player["category_id"] == 1
    assert player["attributes"] == {"role": "player", "team": "left", "jersey": "10"}
    assert player["bbox_pitch"] == {
        "x_bottom_left": 10.0,
        "y_bottom_left": 22.0,
        "x_bottom_middle": 11.0,
        "y_bottom_middle": 22.0,
        "x_bottom_right": 12.0,
        "y_bottom_right": 22.0,
    }
    assert goalkeeper["attributes"] == {
        "role": "goalkeeper",
        "team": "right",
        "jersey": None,
    }
    assert set(goalkeeper["bbox_pitch"].values()) == {-40.0, 5.0}


def test_anonymous_team_is_exported_as_unknown() -> None:
    state = _state()
    state["tracks"]["7"]["team_label"] = "team_0"

    prediction = build_prediction_document(state, _labels())["predictions"][0]

    assert prediction["attributes"]["team"] is None


def test_uncalibrated_observations_produce_valid_empty_predictions() -> None:
    state = _state()
    for observation in state["observations"]:
        observation["pitch_xy"] = None

    assert build_prediction_document(state, _labels())["predictions"] == []


def test_rejects_sequence_frame_count_mismatch() -> None:
    state = _state()
    state["meta"]["frame_count"] = 3

    with pytest.raises(ValueError, match="state has 3 frames but labels contain 2 images"):
        build_prediction_document(state, _labels())


def test_rejects_duplicate_track_identity_in_one_frame() -> None:
    state = _state()
    duplicate = deepcopy(state["observations"][1])
    duplicate["pitch_xy"] = None
    state["observations"].append(duplicate)

    with pytest.raises(ValueError, match="track_id 7 occurs more than once in frame 0"):
        build_prediction_document(state, _labels())


def test_rejects_observation_without_canonical_track() -> None:
    state = _state()
    state["observations"][0]["track_id"] = 99

    with pytest.raises(ValueError, match="references missing track 99"):
        build_prediction_document(state, _labels())


def test_exports_compact_prediction_file(tmp_path: Path) -> None:
    state_path = tmp_path / "gsr.json"
    labels_path = tmp_path / "Labels-GameState.json"
    output_path = tmp_path / "trackers" / "SNGS-001.json"
    state_path.write_text(json.dumps(_state()), encoding="utf-8")
    labels_path.write_text(json.dumps(_labels()), encoding="utf-8")

    result = export_prediction(
        state_path=state_path,
        labels_path=labels_path,
        output_path=output_path,
    )

    assert result == output_path
    assert len(json.loads(output_path.read_text(encoding="utf-8"))["predictions"]) == 2
