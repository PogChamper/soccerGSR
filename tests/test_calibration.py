from __future__ import annotations

import numpy as np

from app.services.calibration import PnLCalibrator
from app.services.clip_state import ClipMeta, ClipState, FrameInfo, FrameObservation


def _shift(dx: float) -> np.ndarray:
    homography = np.eye(3)
    homography[0, 2] = dx
    return homography


def _make_state(solves: dict[int, np.ndarray], n_frames: int) -> tuple[ClipState, PnLCalibrator]:
    """State with one observation per frame; frame i solves to solves[i]."""
    frames = [FrameInfo(frame_idx=index, width=100, height=50) for index in range(n_frames)]
    for index, homography in solves.items():
        frames[index].keypoints = {"h": homography}
    observations = [
        FrameObservation(
            frame_idx=index,
            bbox_xyxy=(0, 0, 20, 20),
            cls_id=0,
            det_confidence=0.9,
            track_id=1,
            foot_xy_image=(10.0, 20.0),
        )
        for index in range(n_frames)
    ]
    state = ClipState(
        meta=ClipMeta("clip.mp4", 100, 50, 25.0, n_frames, n_frames / 25.0, 0.1),
        frames=frames,
        observations=observations,
    )
    calibrator = PnLCalibrator(100, 50)
    calibrator.calibrate_one = lambda keypoints, lines: keypoints.get("h")
    return state, calibrator


def test_calibrate_one_passes_through_the_solver_homography() -> None:
    class Camera:
        def update(self, keypoints, lines) -> None:
            pass

        def heuristic_voting_ground(self, *, refine_lines):
            return {"homography": np.diag([2.0, 3.0, 1.0]), "rep_err": 3.0}

    calibrator = PnLCalibrator(100, 50)
    calibrator._camera = Camera()

    np.testing.assert_allclose(calibrator.calibrate_one({}, {}), np.diag([2.0, 3.0, 1.0]))


def test_calibrate_one_rejects_high_reprojection_error() -> None:
    class Camera:
        def update(self, keypoints, lines) -> None:
            pass

        def heuristic_voting_ground(self, *, refine_lines):
            return {"homography": np.eye(3), "rep_err": 100.0}

    calibrator = PnLCalibrator(100, 50)
    calibrator._camera = Camera()

    assert calibrator.calibrate_one({}, {}) is None


def test_perspective_homography_round_trips_through_the_quad() -> None:
    import cv2

    from app.services.calibration import _image_quad

    world = np.array([(-30.0, -10.0), (30.0, -10.0), (18.0, 25.0), (-18.0, 25.0)])
    homography = cv2.findHomography(_image_quad(100, 50), world)[0]
    state, calibrator = _make_state({index: homography for index in range(8)}, 8)
    for observation in state.observations:
        observation.foot_xy_image = (50.0, 40.0)

    calibrator.calibrate(state)

    foot = homography @ np.array([50.0, 40.0, 1.0])
    expected = foot[:2] / foot[2]
    for observation in state.observations:
        np.testing.assert_allclose(observation.pitch_xy, expected, atol=1e-6)


def test_hold_is_bounded_and_long_gaps_stay_uncalibrated() -> None:
    solves = {index: np.eye(3) for index in [*range(5), *range(25, 30)]}
    state, calibrator = _make_state(solves, 30)

    calibrator.calibrate(state)

    sources = [frame.homography_source for frame in state.frames]
    assert sources == ["solved"] * 5 + ["held"] * 6 + ["none"] * 8 + ["held"] * 6 + ["solved"] * 5
    for frame, observation in zip(state.frames, state.observations):
        if frame.homography_source == "none":
            assert frame.homography_image_to_world is None
            assert observation.pitch_xy is None
        else:
            np.testing.assert_allclose(observation.pitch_xy, (10.0, 20.0), atol=1e-6)

    json_frames = state.to_gsr_json()["frames"]
    assert json_frames[7]["has_calibration"] is True
    assert json_frames[7]["h_source"] == "held"
    assert json_frames[12]["has_calibration"] is False
    assert json_frames[12]["h_source"] == "none"


def test_short_gap_interpolates_between_solves() -> None:
    solves = {index: np.eye(3) for index in range(5)}
    solves |= {index: _shift(2.0) for index in range(10, 15)}
    state, calibrator = _make_state(solves, 15)

    calibrator.calibrate(state)

    assert [frame.homography_source for frame in state.frames[5:10]] == ["interp"] * 5
    np.testing.assert_allclose(state.observations[7].pitch_xy, (11.0, 20.0), atol=1e-6)


def test_solves_survive_a_dropout_during_a_pan() -> None:
    solves = {index: _shift(0.2 * index) for index in [*range(10), *range(20, 31)]}
    state, calibrator = _make_state(solves, 31)

    calibrator.calibrate(state)

    sources = [frame.homography_source for frame in state.frames]
    assert sources == ["solved"] * 10 + ["interp"] * 10 + ["solved"] * 11
    np.testing.assert_allclose(state.observations[15].pitch_xy, (13.0, 20.0), atol=1e-6)
    np.testing.assert_allclose(state.observations[25].pitch_xy, (15.0, 20.0), atol=1e-6)


def test_gate_rejects_flyaway_and_mirrored_solves() -> None:
    solves = {index: np.eye(3) for index in range(11)}
    solves[4] = _shift(50.0)
    solves[7] = np.diag([-1.0, 1.0, 1.0])
    state, calibrator = _make_state(solves, 11)

    calibrator.calibrate(state)

    assert state.frames[4].homography_source == "interp"
    assert state.frames[7].homography_source == "interp"
    for observation in state.observations:
        np.testing.assert_allclose(observation.pitch_xy, (10.0, 20.0), atol=1e-6)


def test_persistent_jump_reseeds_without_interpolating_across_the_cut() -> None:
    solves = {index: np.eye(3) for index in range(5)}
    solves |= {index: _shift(30.0) for index in range(5, 20)}
    state, calibrator = _make_state(solves, 20)

    calibrator.calibrate(state)

    sources = [frame.homography_source for frame in state.frames]
    assert sources == ["solved"] * 5 + ["held"] * 3 + ["none"] + ["held"] * 2 + ["solved"] * 9
    for observation in state.observations[:8]:
        np.testing.assert_allclose(observation.pitch_xy, (10.0, 20.0), atol=1e-6)
    assert state.observations[8].pitch_xy is None
    for observation in state.observations[9:]:
        np.testing.assert_allclose(observation.pitch_xy, (40.0, 20.0), atol=1e-6)
