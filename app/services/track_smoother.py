"""Pitch-space filtering for consolidated tracks: ball teleport gate,
trajectory smoothing, short-gap interpolation."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from app.services.clip_state import ClipState, FrameObservation

logger = logging.getLogger(__name__)

MAX_BALL_SPEED_M_PER_FRAME = 4.0
BALL_RESEED_AFTER = 4
BALL_LOST_GAP_FRAMES = 12

SMOOTH_WINDOW = 13  # 0.43 s at 30 fps
SMOOTH_POLYORDER = 2
SMOOTH_MIN_RUN = 5
SMOOTH_GAP_RESET = 8


def _is_ball(state: ClipState, observation: FrameObservation) -> bool:
    track = state.tracks.get(observation.track_id) if observation.track_id is not None else None
    return (track.cls_id if track is not None else observation.cls_id) == 3


def filter_ball_outliers(state: ClipState) -> int:
    """Drop ball points that teleport; the renderer simply skips them.

    A jump that persists BALL_RESEED_AFTER points in a row is accepted as the
    new position (the ball really was rediscovered elsewhere), as is any point
    after a BALL_LOST_GAP_FRAMES absence.
    """
    ball = sorted(
        (o for o in state.observations if o.pitch_xy is not None and _is_ball(state, o)),
        key=lambda o: o.frame_idx,
    )
    if len(ball) < 2:
        return 0

    last = ball[0]
    reject_run = 0
    dropped = 0
    for observation in ball[1:]:
        gap = observation.frame_idx - last.frame_idx
        distance = float(
            np.hypot(
                observation.pitch_xy[0] - last.pitch_xy[0],
                observation.pitch_xy[1] - last.pitch_xy[1],
            )
        )
        if gap >= BALL_LOST_GAP_FRAMES or distance / max(gap, 1) <= MAX_BALL_SPEED_M_PER_FRAME:
            last = observation
            reject_run = 0
        elif reject_run + 1 >= BALL_RESEED_AFTER:
            last = observation
            reject_run = 0
        else:
            observation.pitch_xy = None
            reject_run += 1
            dropped += 1

    logger.info("ball filter: dropped=%d/%d", dropped, len(ball))
    return dropped


def smooth_pitch_xy(state: ClipState) -> int:
    """Median-filter and Savitzky-Golay person pitch trajectories.

    The 5-point median removes isolated teleports (the outlier resistance the
    polynomial fit lacks) without touching real motion; runs split on gaps
    over SMOOTH_GAP_RESET frames so the fit never bridges an absence. The
    ball is handled by ``filter_ball_outliers`` instead.
    """
    by_track: dict[int, list[FrameObservation]] = defaultdict(list)
    for observation in state.observations:
        if observation.track_id is None or observation.pitch_xy is None or observation.synthetic:
            continue
        track = state.tracks.get(observation.track_id)
        if track is None or track.cls_id == 3:
            continue
        by_track[observation.track_id].append(observation)

    smoothed = 0
    for observations in by_track.values():
        observations.sort(key=lambda item: item.frame_idx)
        run: list[FrameObservation] = []
        for observation in observations:
            if run and observation.frame_idx - run[-1].frame_idx > SMOOTH_GAP_RESET:
                smoothed += _smooth_run(run)
                run = []
            run.append(observation)
        smoothed += _smooth_run(run)

    logger.info("pitch smoothing: smoothed=%d observations", smoothed)
    return smoothed


def _smooth_run(run: list[FrameObservation]) -> int:
    if len(run) < SMOOTH_MIN_RUN:
        return 0
    points = np.array([observation.pitch_xy for observation in run])
    points = median_filter(points, size=(5, 1), mode="mirror")
    window = min(SMOOTH_WINDOW, len(run) if len(run) % 2 else len(run) - 1)
    points = savgol_filter(points, window, SMOOTH_POLYORDER, axis=0, mode="interp")
    for observation, point in zip(run, points):
        observation.pitch_xy = float(point[0]), float(point[1])
    return len(run)


def interpolate_track_gaps(state: ClipState, *, max_frame_delta: int = 10) -> int:
    """Interpolate person observations whose endpoint delta is bounded.

    Frames the calibration left uncalibrated get no synthetic points: the
    renderer freezes there, and a lone fabricated marker would unfreeze it
    into a near-empty pitch.
    """
    if max_frame_delta < 2:
        return 0

    uncalibrated = {frame.frame_idx for frame in state.frames if frame.homography_source == "none"}
    observations_by_track: dict[int, list[FrameObservation]] = defaultdict(list)
    for observation in state.observations:
        if observation.track_id is not None:
            observations_by_track[observation.track_id].append(observation)

    added: list[FrameObservation] = []
    for track_id, observations in observations_by_track.items():
        track = state.tracks.get(track_id)
        if track is None or track.cls_id == 3:
            continue
        observations.sort(key=lambda item: item.frame_idx)
        anchors = [
            item for item in observations if item.pitch_xy is not None and not item.synthetic
        ]
        present = {item.frame_idx for item in observations}
        for left, right in zip(anchors, anchors[1:]):
            delta = right.frame_idx - left.frame_idx
            missing = delta - 1
            if missing < 1 or delta > max_frame_delta:
                continue

            left_bbox = np.asarray(left.bbox_xyxy, dtype=np.float64)
            right_bbox = np.asarray(right.bbox_xyxy, dtype=np.float64)
            left_pitch = np.asarray(left.pitch_xy, dtype=np.float64)
            right_pitch = np.asarray(right.pitch_xy, dtype=np.float64)
            for frame_index in range(left.frame_idx + 1, right.frame_idx):
                if frame_index in present or frame_index in uncalibrated:
                    continue
                fraction = (frame_index - left.frame_idx) / delta
                bbox = tuple(
                    float(value) for value in left_bbox + (right_bbox - left_bbox) * fraction
                )
                pitch = left_pitch + (right_pitch - left_pitch) * fraction
                added.append(
                    FrameObservation(
                        frame_idx=frame_index,
                        bbox_xyxy=bbox,
                        cls_id=left.cls_id,
                        det_confidence=0.5,
                        track_id=track_id,
                        foot_xy_image=((bbox[0] + bbox[2]) / 2.0, bbox[3]),
                        pitch_xy=(float(pitch[0]), float(pitch[1])),
                        team_id=left.team_id,
                        synthetic=True,
                    )
                )

    state.observations.extend(added)
    logger.info(
        "gap interpolation: added=%d max_frame_delta=%d",
        len(added),
        max_frame_delta,
    )
    return len(added)
