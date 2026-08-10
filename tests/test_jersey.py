from __future__ import annotations

import numpy as np
import pytest

from app.services.clip_state import ClipMeta, ClipState, FrameObservation, TrackInfo
from app.services.jersey import JerseyRecognizer, JerseyVoteConfig


class _Session:
    def __init__(self, output_factory):
        self.output_factory = output_factory
        self.calls = 0

    def run(self, output_names, inputs):
        self.calls += 1
        batch = next(iter(inputs.values())).shape[0]
        return self.output_factory(batch)


def _recognizer() -> JerseyRecognizer:
    recognizer = JerseyRecognizer.__new__(JerseyRecognizer)
    recognizer.vote_config = JerseyVoteConfig()
    recognizer._gate_input_name = "image"
    recognizer._ocr_input_name = "image"
    recognizer._ocr_output_names = ["tens", "units"]
    recognizer._gate_session = _Session(lambda batch: [np.full((batch,), 10.0, dtype=np.float32)])

    def ocr(batch):
        tens = np.zeros((batch, 10), dtype=np.float32)
        units = np.zeros((batch, 10), dtype=np.float32)
        tens[:, 1] = 10.0
        units[:, 0] = 10.0
        return [tens, units]

    recognizer._ocr_session = _Session(ocr)
    return recognizer


def test_process_boxes_batches_gate_and_ocr() -> None:
    recognizer = _recognizer()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    results = recognizer.process_boxes(
        frame,
        [(0.0, 0.0, 20.0, 40.0), (30.0, 0.0, 50.0, 40.0)],
    )

    assert recognizer._gate_session.calls == 1
    assert recognizer._ocr_session.calls == 1
    assert len(results) == 2
    assert all(result.ocr_logits_tens is not None for result in results)


def test_process_boxes_rejects_invalid_gate_output_shape() -> None:
    recognizer = _recognizer()
    recognizer._gate_session = _Session(lambda batch: [np.zeros((batch, 2), dtype=np.float32)])

    with pytest.raises(ValueError, match="one logit per crop"):
        recognizer.process_boxes(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [(0.0, 0.0, 20.0, 40.0)],
        )


def test_process_boxes_rejects_invalid_ocr_output_shape() -> None:
    recognizer = _recognizer()
    recognizer._ocr_session = _Session(
        lambda batch: [
            np.zeros((batch, 9), dtype=np.float32),
            np.zeros((batch, 10), dtype=np.float32),
        ]
    )

    with pytest.raises(ValueError, match="must both have shape"):
        recognizer.process_boxes(
            np.zeros((100, 100, 3), dtype=np.uint8),
            [(0.0, 0.0, 20.0, 40.0)],
        )


def test_commit_requires_six_accepted_frames() -> None:
    recognizer = _recognizer()
    tens = np.zeros(10, dtype=np.float32)
    units = np.zeros(10, dtype=np.float32)
    tens[1] = 10.0
    units[0] = 10.0

    assert recognizer._commit_from_logits(tens.tolist(), units.tolist(), 5)[0] is None
    assert recognizer._commit_from_logits(tens.tolist(), units.tolist(), 6)[0] == 10


def test_dedup_does_not_mix_tracks_without_a_team() -> None:
    recognizer = _recognizer()
    state = ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, 1, 0.04, 0.1),
        tracks={
            track_id: TrackInfo(
                track_id=track_id,
                cls_id=0,
                cls_name="player",
                jersey_number=10,
                jersey_confidence=0.99,
                jersey_vote_count=6,
            )
            for track_id in (1, 2)
        },
    )

    recognizer.dedup_numbers(state)

    assert [track.jersey_number for track in state.tracks.values()] == [10, 10]


def test_goalkeeper_does_not_receive_a_jersey_attribute() -> None:
    recognizer = _recognizer()
    tens = [0.0] * 10
    units = [0.0] * 10
    tens[1] = 10.0
    units[0] = 10.0
    state = ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, 6, 0.24, 0.1),
        tracks={1: TrackInfo(track_id=1, cls_id=1, cls_name="goalkeeper")},
        observations=[
            FrameObservation(
                frame_idx=frame,
                bbox_xyxy=(0, 0, 10, 20),
                cls_id=1,
                det_confidence=0.9,
                track_id=1,
                visibility_p=0.99,
                ocr_logits_tens=tens,
                ocr_logits_units=units,
            )
            for frame in range(6)
        ],
    )

    recognizer.collect_votes_into_tracks(state)
    recognizer.commit_numbers(state)

    assert state.tracks[1].jersey_number is None
    assert state.tracks[1].jersey_vote_count == 0
