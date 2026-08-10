"""Build SoccerNetGS predictions from a service ``gsr.json`` document."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

CLASS_ROLES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
}

_ORIENTED_TEAMS = {
    "team_left": "left",
    "team_right": "right",
    "goalkeeper_left": "left",
    "goalkeeper_right": "right",
}
_ANONYMOUS_TEAMS = {
    None,
    "team_0",
    "team_1",
    "goalkeeper_0",
    "goalkeeper_1",
}


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _point(value: Any, path: str) -> tuple[float, float]:
    values = _array(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain two coordinates")
    return _number(values[0], f"{path}[0]"), _number(values[1], f"{path}[1]")


def _bbox(value: Any, path: str) -> tuple[float, float, float, float]:
    values = _array(value, path)
    if len(values) != 4:
        raise ValueError(f"{path} must contain four coordinates")
    x1, y1, x2, y2 = (_number(item, f"{path}[{index}]") for index, item in enumerate(values))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{path} must have positive width and height")
    return x1, y1, x2, y2


def _homography(value: Any, path: str) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    rows = _array(value, path)
    if len(rows) != 3:
        raise ValueError(f"{path} must be a 3x3 matrix")
    matrix = []
    for row_index, row in enumerate(rows):
        values = _array(row, f"{path}[{row_index}]")
        if len(values) != 3:
            raise ValueError(f"{path} must be a 3x3 matrix")
        matrix.append(
            tuple(
                _number(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(values)
            )
        )
    return tuple(matrix)


def _project(
    homography: tuple[tuple[float, float, float], ...],
    x: float,
    y: float,
) -> tuple[float, float] | None:
    denominator = homography[2][0] * x + homography[2][1] * y + homography[2][2]
    if abs(denominator) < 1e-12:
        return None
    world_x = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denominator
    world_y = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denominator
    if not math.isfinite(world_x) or not math.isfinite(world_y):
        return None
    return world_x, world_y


def _pitch_bbox(
    bbox: tuple[float, float, float, float],
    center: tuple[float, float],
    homography: tuple[tuple[float, float, float], ...] | None,
) -> dict[str, float]:
    left = middle = right = center
    if homography is not None:
        x1, _, x2, y2 = bbox
        projected = (
            _project(homography, x1, y2),
            _project(homography, (x1 + x2) / 2.0, y2),
            _project(homography, x2, y2),
        )
        if all(point is not None for point in projected):
            raw_left, raw_middle, raw_right = projected
            offset_x = center[0] - raw_middle[0]
            offset_y = center[1] - raw_middle[1]
            left = raw_left[0] + offset_x, raw_left[1] + offset_y
            right = raw_right[0] + offset_x, raw_right[1] + offset_y

    return {
        "x_bottom_left": left[0],
        "y_bottom_left": left[1],
        "x_bottom_middle": middle[0],
        "y_bottom_middle": middle[1],
        "x_bottom_right": right[0],
        "y_bottom_right": right[1],
    }


def _ordered_images(labels: dict[str, Any]) -> list[dict[str, Any]]:
    images = [
        _object(image, f"labels.images[{index}]")
        for index, image in enumerate(_array(labels.get("images"), "labels.images"))
    ]
    if not images:
        raise ValueError("labels.images must not be empty")

    image_ids = []
    frame_numbers = []
    for index, image in enumerate(images):
        image_id = image.get("image_id")
        if not isinstance(image_id, (str, int)) or isinstance(image_id, bool):
            raise ValueError(f"labels.images[{index}].image_id must be a string or integer")
        image_ids.append(str(image_id))
        try:
            frame_numbers.append(int(str(image_id).split("_")[-1]))
        except ValueError as exc:
            raise ValueError(
                "labels image_id values must end with an integer frame number"
            ) from exc
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("labels.images contains duplicate image_id values")
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("labels.images contains duplicate frame numbers")

    return [image for _, image in sorted(zip(frame_numbers, images), key=lambda item: item[0])]


def _category_ids(labels: dict[str, Any]) -> dict[str, int]:
    categories = _array(labels.get("categories"), "labels.categories")
    result: dict[str, int] = {}
    for index, value in enumerate(categories):
        category = _object(value, f"labels.categories[{index}]")
        if category.get("supercategory") != "object":
            continue
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"labels.categories[{index}].name must be a non-empty string")
        if name in result:
            raise ValueError(f"labels.categories contains duplicate object category {name!r}")
        result[name] = _integer(category.get("id"), f"labels.categories[{index}].id", minimum=1)
    return result


def _tracks(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    values = _object(state.get("tracks"), "state.tracks")
    result: dict[int, dict[str, Any]] = {}
    for key, value in values.items():
        track = _object(value, f"state.tracks[{key!r}]")
        track_id = _integer(track.get("track_id"), f"state.tracks[{key!r}].track_id", minimum=0)
        try:
            key_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"state.tracks key {key!r} is not an integer") from exc
        if key_id != track_id:
            raise ValueError(f"state.tracks key {key!r} does not match track_id {track_id}")
        if track_id in result:
            raise ValueError(f"state.tracks contains duplicate track_id {track_id}")
        class_id = _integer(track.get("cls_id"), f"state.tracks[{key!r}].cls_id", minimum=0)
        if class_id not in CLASS_ROLES:
            raise ValueError(f"state.tracks[{key!r}].cls_id is not a supported service class")
        result[track_id] = track
    return result


def _team(track: dict[str, Any], role: str, path: str) -> str | None:
    if role not in {"player", "goalkeeper"}:
        return None
    label = track.get("team_label")
    if label is not None and not isinstance(label, str):
        raise ValueError(f"{path}.team_label must be a string or null")
    if label in _ORIENTED_TEAMS:
        return _ORIENTED_TEAMS[label]
    if label in _ANONYMOUS_TEAMS:
        return None
    raise ValueError(f"{path}.team_label is not a supported service team label")


def _jersey(track: dict[str, Any], role: str, path: str) -> str | None:
    value = track.get("jersey_number")
    if role != "player" or value is None:
        return None
    number = _integer(value, f"{path}.jersey_number", minimum=0)
    if number > 10000:
        raise ValueError(f"{path}.jersey_number must not exceed 10000")
    return str(number)


def build_prediction_document(
    service_state: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    """Convert one service result using its sequence label file as frame metadata.

    Observations without pitch coordinates are omitted. Malformed state or
    sequence metadata raises ``ValueError``.
    """

    state = _object(service_state, "state")
    label_data = _object(labels, "labels")
    images = _ordered_images(label_data)
    categories = _category_ids(label_data)

    meta = _object(state.get("meta"), "state.meta")
    frame_count = _integer(meta.get("frame_count"), "state.meta.frame_count", minimum=1)
    width = _integer(meta.get("width"), "state.meta.width", minimum=1)
    height = _integer(meta.get("height"), "state.meta.height", minimum=1)
    if frame_count != len(images):
        raise ValueError(f"state has {frame_count} frames but labels contain {len(images)} images")

    for index, image in enumerate(images):
        image_width = _integer(image.get("width"), f"labels.images[{index}].width", minimum=1)
        image_height = _integer(image.get("height"), f"labels.images[{index}].height", minimum=1)
        if (image_width, image_height) != (width, height):
            raise ValueError(
                f"labels image {image['image_id']!r} is {image_width}x{image_height}; "
                f"state is {width}x{height}"
            )

    frame_values = _array(state.get("frames"), "state.frames")
    frame_homographies: dict[int, tuple[tuple[float, float, float], ...] | None] = {}
    for index, value in enumerate(frame_values):
        frame = _object(value, f"state.frames[{index}]")
        frame_index = _integer(
            frame.get("frame_idx"), f"state.frames[{index}].frame_idx", minimum=0
        )
        if frame_index in frame_homographies:
            raise ValueError(f"state.frames contains duplicate frame_idx {frame_index}")
        frame_width = _integer(frame.get("width"), f"state.frames[{index}].width", minimum=1)
        frame_height = _integer(frame.get("height"), f"state.frames[{index}].height", minimum=1)
        if (frame_width, frame_height) != (width, height):
            raise ValueError(
                f"state.frames[{index}] is {frame_width}x{frame_height}; state is {width}x{height}"
            )
        frame_homographies[frame_index] = _homography(
            frame.get("H_img2world"), f"state.frames[{index}].H_img2world"
        )
    expected_indices = set(range(frame_count))
    if set(frame_homographies) != expected_indices:
        raise ValueError("state.frames must contain every frame_idx exactly once")

    tracks = _tracks(state)
    predictions: list[tuple[int, int, dict[str, Any]]] = []
    observed_ids: set[tuple[int, int]] = set()
    observations = _array(state.get("observations"), "state.observations")
    for index, value in enumerate(observations):
        path = f"state.observations[{index}]"
        observation = _object(value, path)
        frame_index = _integer(observation.get("frame_idx"), f"{path}.frame_idx", minimum=0)
        if frame_index >= frame_count:
            raise ValueError(f"{path}.frame_idx is outside the sequence")
        bbox = _bbox(observation.get("bbox_xyxy"), f"{path}.bbox_xyxy")
        confidence = _number(observation.get("det_confidence"), f"{path}.det_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{path}.det_confidence must be between 0 and 1")
        track_id_value = observation.get("track_id")
        if track_id_value is None:
            continue
        track_id = _integer(track_id_value, f"{path}.track_id", minimum=0)
        if track_id not in tracks:
            raise ValueError(f"{path}.track_id references missing track {track_id}")
        identity = frame_index, track_id
        if identity in observed_ids:
            raise ValueError(f"track_id {track_id} occurs more than once in frame {frame_index}")
        observed_ids.add(identity)

        track = tracks[track_id]
        class_id = track["cls_id"]
        observation_class = _integer(observation.get("cls_id"), f"{path}.cls_id", minimum=0)
        if observation_class != class_id:
            raise ValueError(
                f"{path}.cls_id {observation_class} does not match track class {class_id}"
            )
        role = CLASS_ROLES[class_id]
        if role not in categories:
            raise ValueError(f"labels.categories has no object category for role {role!r}")

        pitch_value = observation.get("pitch_xy")
        if pitch_value is None:
            continue
        pitch_center = _point(pitch_value, f"{path}.pitch_xy")
        prediction = {
            "image_id": images[frame_index]["image_id"],
            "track_id": track_id,
            "supercategory": "object",
            "category_id": categories[role],
            "confidence": confidence,
            "bbox_pitch": _pitch_bbox(
                bbox,
                pitch_center,
                frame_homographies[frame_index],
            ),
            "attributes": {
                "role": role,
                "team": _team(track, role, f"state.tracks[{track_id}]"),
                "jersey": _jersey(track, role, f"state.tracks[{track_id}]"),
            },
        }
        predictions.append((frame_index, track_id, prediction))

    predictions.sort(key=lambda item: (item[0], item[1]))
    return {
        "images": deepcopy(_array(label_data.get("images"), "labels.images")),
        "categories": deepcopy(_array(label_data.get("categories"), "labels.categories")),
        "predictions": [prediction for _, _, prediction in predictions],
    }


def export_prediction(
    *,
    state_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Read one service result and write one TrackEval prediction file."""

    state_file = Path(state_path)
    labels_file = Path(labels_path)
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in service state: {state_file}") from exc
    try:
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in sequence labels: {labels_file}") from exc

    document = build_prediction_document(state, labels)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return destination
