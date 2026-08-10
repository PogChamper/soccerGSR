"""Batched jersey recognition and per-identity temporal voting."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import get_settings
from app.services.clip_state import ClipState
from app.utils.cuda_env import get_providers
from app.utils.models_registry import ensure_model

logger = logging.getLogger(__name__)

_GATE_IMAGE_SIZE = 128
_OCR_IMAGE_SIZE = 224
_DIGIT_CLASSES = 10
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class JerseyVoteConfig:
    """Thresholds validated on the GSR validation split."""

    visibility_threshold: float = 0.7
    min_digit_confidence: float = 0.9
    min_votes: int = 6


@dataclass(slots=True)
class JerseyResult:
    """Visibility and optional two-head OCR output for one crop."""

    visibility_p: float
    ocr_logits_tens: list[float] | None = None
    ocr_logits_units: list[float] | None = None


def _crop_bgr(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(x2))
    bottom = min(height, int(y2))
    if right - left < 6 or bottom - top < 12:
        return None
    return frame[top:bottom, left:right]


def _preprocess(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    """Convert a BGR crop to an ImageNet-normalized CHW tensor."""
    image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = image.astype(np.float32) / 255.0
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor.transpose(2, 0, 1)


def _make_batch(crops_bgr: Sequence[np.ndarray], size: int) -> np.ndarray:
    return np.stack([_preprocess(crop, size) for crop in crops_bgr])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


class JerseyRecognizer:
    """Run the visibility gate, OCR, and temporal vote aggregation."""

    def __init__(self, vote_config: JerseyVoteConfig | None = None) -> None:
        import onnxruntime as ort

        settings = get_settings()
        gate_path = ensure_model(
            "visibility_gate",
            auto_download=settings.model_auto_download,
        )
        ocr_path = ensure_model(
            "jersey_ocr",
            auto_download=settings.model_auto_download,
        )

        providers = get_providers(prefer_gpu=True)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self._gate_session = ort.InferenceSession(
            str(gate_path),
            sess_options=session_options,
            providers=providers,
        )
        self._ocr_session = ort.InferenceSession(
            str(ocr_path),
            sess_options=session_options,
            providers=providers,
        )

        gate_inputs = self._gate_session.get_inputs()
        gate_outputs = self._gate_session.get_outputs()
        if len(gate_inputs) != 1 or len(gate_outputs) != 1:
            raise ValueError("visibility gate ONNX must expose one input and one output")

        ocr_inputs = self._ocr_session.get_inputs()
        ocr_outputs = self._ocr_session.get_outputs()
        if len(ocr_inputs) != 1 or len(ocr_outputs) != 2:
            raise ValueError("jersey OCR ONNX must expose one input and two outputs")

        self._gate_input_name = gate_inputs[0].name
        self._ocr_input_name = ocr_inputs[0].name
        self._ocr_output_names = [output.name for output in ocr_outputs]
        self.vote_config = vote_config or JerseyVoteConfig()
        logger.info(
            "Jersey models loaded: gate=%s ocr=%s providers=%s",
            gate_path.name,
            ocr_path.name,
            self._gate_session.get_providers(),
        )

    def process_boxes(
        self,
        frame: np.ndarray,
        boxes: Sequence[tuple[float, float, float, float]],
    ) -> list[JerseyResult]:
        """Recognize a batch with at most one gate and one OCR call."""
        results = [JerseyResult(visibility_p=0.0) for _ in boxes]
        crops: list[np.ndarray] = []
        result_indices: list[int] = []
        for result_index, box in enumerate(boxes):
            crop = _crop_bgr(frame, box)
            if crop is not None:
                crops.append(crop)
                result_indices.append(result_index)
        if not crops:
            return results

        gate_input = _make_batch(crops, _GATE_IMAGE_SIZE)
        gate_output = self._gate_session.run(
            None,
            {self._gate_input_name: gate_input},
        )[0]
        gate_logits = np.asarray(gate_output, dtype=np.float32)
        if gate_logits.size != len(crops):
            raise ValueError(
                "visibility gate output must contain one logit per crop, "
                f"got shape {gate_logits.shape} for batch {len(crops)}"
            )

        visibility = _sigmoid(gate_logits.reshape(-1))
        visible_crop_indices: list[int] = []
        for crop_index, result_index in enumerate(result_indices):
            probability = float(visibility[crop_index])
            results[result_index].visibility_p = probability
            if probability >= self.vote_config.visibility_threshold:
                visible_crop_indices.append(crop_index)

        if not visible_crop_indices:
            return results

        ocr_input = _make_batch(
            [crops[index] for index in visible_crop_indices],
            _OCR_IMAGE_SIZE,
        )
        ocr_outputs = self._ocr_session.run(
            self._ocr_output_names,
            {self._ocr_input_name: ocr_input},
        )
        tens = np.asarray(ocr_outputs[0], dtype=np.float32)
        units = np.asarray(ocr_outputs[1], dtype=np.float32)
        expected_shape = (len(visible_crop_indices), _DIGIT_CLASSES)
        if tens.shape != expected_shape or units.shape != expected_shape:
            raise ValueError(
                "jersey OCR outputs must both have shape "
                f"{expected_shape}, got {tens.shape} and {units.shape}"
            )

        for output_index, crop_index in enumerate(visible_crop_indices):
            result = results[result_indices[crop_index]]
            result.ocr_logits_tens = tens[output_index].tolist()
            result.ocr_logits_units = units[output_index].tolist()
        return results

    def _commit_from_logits(
        self,
        logits_tens: list[float] | None,
        logits_units: list[float] | None,
        vote_count: int,
    ) -> tuple[int | None, float | None, str]:
        """Decode pooled OCR logits or return a rejection reason."""
        if logits_tens is None or logits_units is None:
            return None, None, "no_votes"
        if vote_count < self.vote_config.min_votes:
            return None, None, "too_few"

        tens_sum = np.asarray(logits_tens, dtype=np.float32)
        units_sum = np.asarray(logits_units, dtype=np.float32)
        tens_digit = int(np.argmax(tens_sum))
        units_digit = int(np.argmax(units_sum))
        number = units_digit if tens_digit == 0 else tens_digit * 10 + units_digit
        tens_probability = _softmax(tens_sum / vote_count)[tens_digit]
        units_probability = _softmax(units_sum / vote_count)[units_digit]
        confidence = float(min(tens_probability, units_probability))
        return number, confidence, "ok"

    def collect_votes_into_tracks(self, state: ClipState) -> None:
        """Accumulate accepted OCR logits on each raw player fragment."""
        observations_by_track = state.observations_by_track()
        total_observations = 0
        visible_observations = 0
        accepted_observations = 0

        for track_id, observations in observations_by_track.items():
            track = state.tracks.get(track_id)
            if track is None or track.cls_id != 0:
                continue

            tens_sum = np.zeros(_DIGIT_CLASSES, dtype=np.float32)
            units_sum = np.zeros(_DIGIT_CLASSES, dtype=np.float32)
            visible_count = 0
            vote_count = 0
            for observation in observations:
                total_observations += 1
                if (
                    observation.visibility_p is None
                    or observation.visibility_p < self.vote_config.visibility_threshold
                    or observation.ocr_logits_tens is None
                    or observation.ocr_logits_units is None
                ):
                    continue

                visible_observations += 1
                visible_count += 1
                tens_logits = np.asarray(observation.ocr_logits_tens, dtype=np.float32)
                units_logits = np.asarray(observation.ocr_logits_units, dtype=np.float32)
                tens_probabilities = _softmax(tens_logits)
                units_probabilities = _softmax(units_logits)
                tens_digit = int(np.argmax(tens_probabilities))
                units_digit = int(np.argmax(units_probabilities))
                digit_confidence = float(
                    min(tens_probabilities[tens_digit], units_probabilities[units_digit])
                )
                if digit_confidence < self.vote_config.min_digit_confidence:
                    continue

                accepted_observations += 1
                tens_sum += tens_logits
                units_sum += units_logits
                vote_count += 1

            track.jersey_logits_tens = tens_sum.tolist() if vote_count else None
            track.jersey_logits_units = units_sum.tolist() if vote_count else None
            track.jersey_vote_count = vote_count
            track.n_frames_visible_gate = visible_count
            number, confidence, _ = self._commit_from_logits(
                track.jersey_logits_tens,
                track.jersey_logits_units,
                vote_count,
            )
            track.jersey_number = number
            track.jersey_confidence = confidence

        logger.info(
            "Jersey votes collected: observations=%d visible=%d accepted=%d fragments=%d",
            total_observations,
            visible_observations,
            accepted_observations,
            len(observations_by_track),
        )

    def commit_numbers(self, state: ClipState) -> None:
        """Decode pooled evidence after fragment merging."""
        committed = 0
        reasons: dict[str, int] = {}
        for track in state.tracks.values():
            if track.cls_id != 0:
                continue
            number, confidence, status = self._commit_from_logits(
                track.jersey_logits_tens,
                track.jersey_logits_units,
                track.jersey_vote_count,
            )
            reasons[status] = reasons.get(status, 0) + 1
            track.jersey_number = number
            track.jersey_confidence = confidence
            if number is not None:
                committed += 1

        logger.info(
            "Jersey numbers committed: committed=%d tracks=%d reasons=%s",
            committed,
            len(state.tracks),
            reasons,
        )

    def dedup_numbers(self, state: ClipState) -> None:
        """Keep one claim for each team and jersey number."""
        by_team_number: dict[tuple[int, int], list[int]] = {}
        for track_id, track in state.tracks.items():
            if track.jersey_number is None or track.team_id is None:
                continue
            key = (track.team_id, track.jersey_number)
            by_team_number.setdefault(key, []).append(track_id)

        def evidence(track_id: int) -> float:
            track = state.tracks[track_id]
            return float(track.jersey_vote_count) * float(track.jersey_confidence or 0.0)

        dropped = 0
        for (team_id, number), track_ids in by_team_number.items():
            if len(track_ids) <= 1:
                continue
            ranked_track_ids = sorted(track_ids, key=evidence, reverse=True)
            keeper_id = ranked_track_ids[0]
            keeper_evidence = evidence(keeper_id)
            for duplicate_id in ranked_track_ids[1:]:
                duplicate = state.tracks[duplicate_id]
                logger.debug(
                    "Dropping duplicate jersey claim: track=%d jersey=%d team=%d "
                    "evidence=%.1f keeper=%d keeper_evidence=%.1f",
                    duplicate_id,
                    number,
                    team_id,
                    evidence(duplicate_id),
                    keeper_id,
                    keeper_evidence,
                )
                duplicate.jersey_number = None
                duplicate.jersey_confidence = None
                dropped += 1

        if dropped:
            logger.info("Duplicate jersey claims dropped: count=%d", dropped)


_INSTANCE: JerseyRecognizer | None = None


def get_jersey_recognizer() -> JerseyRecognizer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = JerseyRecognizer()
    return _INSTANCE
