from __future__ import annotations

from app.services.clip_state import ClipMeta, ClipState, FrameObservation, TrackInfo


def test_json_separates_effective_and_raw_detection_class() -> None:
    state = ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, 1, 0.04, 0.1),
        observations=[
            FrameObservation(
                frame_idx=0,
                bbox_xyxy=(0, 0, 10, 20),
                cls_id=0,
                det_confidence=0.9,
                track_id=1,
            )
        ],
        tracks={1: TrackInfo(track_id=1, cls_id=2, cls_name="referee")},
    )

    observation = state.to_gsr_json()["observations"][0]

    assert observation["cls_id"] == 2
    assert observation["raw_cls_id"] == 0
