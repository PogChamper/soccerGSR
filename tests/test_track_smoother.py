from __future__ import annotations

from app.services.clip_state import ClipMeta, ClipState, FrameInfo, FrameObservation, TrackInfo
from app.services.track_smoother import (
    filter_ball_outliers,
    interpolate_track_gaps,
    smooth_pitch_xy,
)


def _state(last_frame: int) -> ClipState:
    return ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, 20, 0.8, 0.1),
        observations=[
            FrameObservation(
                frame_idx=0,
                bbox_xyxy=(0, 0, 10, 20),
                cls_id=0,
                det_confidence=0.9,
                track_id=1,
                pitch_xy=(5.0, 20.0),
            ),
            FrameObservation(
                frame_idx=last_frame,
                bbox_xyxy=(last_frame, 0, 10 + last_frame, 20),
                cls_id=0,
                det_confidence=0.9,
                track_id=1,
                pitch_xy=(5.0 + last_frame, 20.0),
            ),
        ],
        tracks={1: TrackInfo(track_id=1, cls_id=0, cls_name="player")},
    )


def _player_state(positions: dict[int, tuple[float, float]]) -> ClipState:
    return ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, len(positions), 0.8, 0.1),
        observations=[
            FrameObservation(
                frame_idx=index,
                bbox_xyxy=(0, 0, 10, 20),
                cls_id=0,
                det_confidence=0.9,
                track_id=1,
                pitch_xy=position,
            )
            for index, position in sorted(positions.items())
        ],
        tracks={1: TrackInfo(track_id=1, cls_id=0, cls_name="player")},
    )


def test_interpolation_fills_endpoint_delta_of_ten_frames() -> None:
    state = _state(10)

    added = interpolate_track_gaps(state, max_frame_delta=10)

    assert added == 9
    assert sum(observation.synthetic for observation in state.observations) == 9
    middle = next(observation for observation in state.observations if observation.frame_idx == 5)
    assert middle.bbox_xyxy == (5.0, 0.0, 15.0, 20.0)
    assert middle.pitch_xy == (10.0, 20.0)
    assert middle.det_confidence == 0.5


def test_interpolation_rejects_endpoint_delta_over_ten_frames() -> None:
    state = _state(11)

    assert interpolate_track_gaps(state, max_frame_delta=10) == 0


def test_interpolation_skips_uncalibrated_frames() -> None:
    state = _state(2)
    state.frames = [FrameInfo(frame_idx=1, width=100, height=100)]

    assert interpolate_track_gaps(state, max_frame_delta=10) == 0

    state.frames[0].homography_source = "held"

    assert interpolate_track_gaps(state, max_frame_delta=10) == 1


def test_smooth_pitch_xy_removes_spike_and_keeps_motion() -> None:
    positions = {index: (0.1 * index, 20.0) for index in range(20)}
    positions[10] = (30.0, 20.0)
    state = _player_state(positions)

    assert smooth_pitch_xy(state) == 20

    xs = [observation.pitch_xy[0] for observation in state.observations]
    assert abs(xs[10] - 1.0) < 0.1
    assert abs(xs[17] - xs[3] - 1.4) < 0.1
    assert all(abs(observation.pitch_xy[1] - 20.0) < 1e-9 for observation in state.observations)


def test_smooth_pitch_xy_survives_an_outlier_at_run_start() -> None:
    positions = {index: (0.0, 0.0) for index in range(20)}
    positions[0] = (20.0, 0.0)
    state = _player_state(positions)

    smooth_pitch_xy(state)

    assert max(abs(observation.pitch_xy[0]) for observation in state.observations) < 0.05


def test_smooth_pitch_xy_keeps_real_motion_of_a_sparse_track() -> None:
    positions = {4 * index: (2.0 * index, 0.0) for index in range(10)}
    state = _player_state(positions)

    smooth_pitch_xy(state)

    xs = [observation.pitch_xy[0] for observation in state.observations]
    assert xs[-1] - xs[0] > 14.0


def _ball_state(positions: list[tuple[float, float]]) -> ClipState:
    return ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, len(positions), 0.8, 0.1),
        observations=[
            FrameObservation(
                frame_idx=index,
                bbox_xyxy=(0, 0, 5, 5),
                cls_id=3,
                det_confidence=0.7,
                pitch_xy=position,
            )
            for index, position in enumerate(positions)
        ],
    )


def test_ball_filter_drops_teleports() -> None:
    positions = [(0.5 * index, 0.0) for index in range(8)]
    positions[4] = (40.0, 0.0)
    state = _ball_state(positions)

    assert filter_ball_outliers(state) == 1
    assert state.observations[4].pitch_xy is None
    assert state.observations[5].pitch_xy == (2.5, 0.0)


def test_ball_filter_reseeds_after_persistent_jump() -> None:
    positions = [(0.1 * index, 0.0) for index in range(3)]
    positions += [(50.0 + 0.1 * index, 0.0) for index in range(5)]
    state = _ball_state(positions)

    assert filter_ball_outliers(state) == 3
    assert all(observation.pitch_xy is None for observation in state.observations[3:6])
    assert state.observations[6].pitch_xy is not None
    assert state.observations[7].pitch_xy is not None


def test_existing_observation_is_not_resynthesized() -> None:
    state = _state(2)
    state.observations.append(
        FrameObservation(
            frame_idx=1,
            bbox_xyxy=(1, 0, 11, 20),
            cls_id=3,
            det_confidence=0.7,
            track_id=1,
        )
    )

    added = interpolate_track_gaps(state, max_frame_delta=10)

    assert added == 0
    keys = [(observation.frame_idx, observation.track_id) for observation in state.observations]
    assert len(keys) == len(set(keys))
