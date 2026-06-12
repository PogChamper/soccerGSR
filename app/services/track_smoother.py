"""Post-calibration cleanup of pitch coordinates, applied in place:
ball teleport filter, per-track EMA smoothing of ``pitch_xy`` (bbox foot and
homography are both noisy — ~0.5-1 m of jitter in the far half-pitch),
space-time class voting and short-gap interpolation for the minimap.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from app.services.clip_state import ClipState, FrameObservation

logger = logging.getLogger(__name__)


DEFAULT_ALPHA = 0.22             # 0.0=frozen, 1.0=no smoothing (per pass)
DEFAULT_GAP_RESET_FRAMES = 8
# Physically a player cannot move more than ~10 m/s, i.e. ~0.33 m at 30 fps.
# A jump much bigger than that is almost certainly a calibration outlier
# (e.g. PnLCalib switching to another candidate H). For outlier frames we
# trust the smoothed state more and rely on the EMA pulling back to truth.
OUTLIER_DELTA_M = 3.0
OUTLIER_ALPHA = 0.08
# "Dead-band": if the smoothed position would move less than this between
# consecutive frames, freeze it. Kills the residual visible jitter for
# players who are actually standing still.
DEAD_BAND_M = 0.16


# ---- ball outlier rejection ------------------------------------------------

# A real ball tops out around ~36 m/s; at 25 fps that is ~1.5 m/frame. We allow
# a generous margin and reject anything above this as a false detection (a sock,
# a line, crowd) that got projected to a wild pitch location.
MAX_BALL_SPEED_M_PER_FRAME = 4.0
BALL_RESEED_AFTER = 4          # consecutive rejects -> accept (ball truly moved)
BALL_LOST_GAP_FRAMES = 12      # gap this long -> redetection elsewhere is legit


def filter_ball_outliers(state: ClipState) -> int:
    """Drop physically-impossible ball jumps (teleports) in-place.

    Sets ``pitch_xy = None`` on ball observations that imply an impossible
    speed from the last accepted ball position, so the renderer simply skips
    them instead of flashing the ball across the pitch.
    """
    ball = sorted(
        (o for o in state.observations if o.cls_id == 3 and o.pitch_xy is not None),
        key=lambda o: o.frame_idx,
    )
    if len(ball) < 2:
        return 0

    last_f = ball[0].frame_idx
    last_xy = ball[0].pitch_xy
    reject_run = 0
    n_drop = 0
    for o in ball[1:]:
        df = o.frame_idx - last_f
        dist = ((o.pitch_xy[0] - last_xy[0]) ** 2
                + (o.pitch_xy[1] - last_xy[1]) ** 2) ** 0.5
        speed = dist / max(df, 1)
        if df >= BALL_LOST_GAP_FRAMES or speed <= MAX_BALL_SPEED_M_PER_FRAME:
            last_f, last_xy = o.frame_idx, o.pitch_xy
            reject_run = 0
        else:
            reject_run += 1
            if reject_run >= BALL_RESEED_AFTER:
                last_f, last_xy = o.frame_idx, o.pitch_xy
                reject_run = 0
            else:
                o.pitch_xy = None
                n_drop += 1

    logger.info(f"ball filter: dropped {n_drop}/{len(ball)} teleporting ball points")
    return n_drop


# ---- space-time detection-class voting ------------------------------------

# Vote each detection's class over a small space-time tube (pooled by
# location, not by track id) — removes class flicker when the tracker swaps
# ids on a referee in dark kit or a far goalkeeper.
CLASS_VOTE_WINDOW = 8          # +/- frames
CLASS_VOTE_RADIUS_PX = 55      # screen-pixel neighbourhood


def smooth_detection_class(
    state: ClipState,
    *,
    window: int = CLASS_VOTE_WINDOW,
    radius_px: float = CLASS_VOTE_RADIUS_PX,
) -> int:
    """Vote each detection's display class over a space-time tube.

    Writes ``obs.display_cls`` for every real (non-synthetic) observation.
    Distance-weighted so a detection's own track (≈0 px across frames)
    dominates over other objects that pass nearby. Ball (cls 3) is left as-is.

    Returns the number of observations whose voted class differs from the raw
    detector class.
    """
    obs_by_frame: Dict[int, List[FrameObservation]] = defaultdict(list)
    targets: List[FrameObservation] = []
    for o in state.observations:
        if o.synthetic:
            continue
        if o.cls_id == 3:                      # ball keeps its class
            o.display_cls = 3
            continue
        cx = (o.bbox_xyxy[0] + o.bbox_xyxy[2]) * 0.5
        cy = (o.bbox_xyxy[1] + o.bbox_xyxy[3]) * 0.5
        obs_by_frame[o.frame_idx].append(o)
        targets.append(o)

    # cache centres
    centre = {
        id(o): ((o.bbox_xyxy[0] + o.bbox_xyxy[2]) * 0.5,
                (o.bbox_xyxy[1] + o.bbox_xyxy[3]) * 0.5)
        for o in targets
    }

    n_changed = 0
    r2 = radius_px * radius_px
    for o in targets:
        ox, oy = centre[id(o)]
        votes: Dict[int, float] = defaultdict(float)
        for f in range(o.frame_idx - window, o.frame_idx + window + 1):
            for n in obs_by_frame.get(f, ()):
                nx, ny = centre[id(n)]
                d2 = (nx - ox) ** 2 + (ny - oy) ** 2
                if d2 > r2:
                    continue
                w = (n.det_confidence or 0.1) / (1.0 + (d2 ** 0.5) / 10.0)
                votes[n.cls_id] += w
        if votes:
            o.display_cls = max(votes, key=votes.get)
        else:
            o.display_cls = o.cls_id

    # Stage 2: per-track temporal majority over the voted labels. Collapses
    # isolated single-frame flips that survive the spatial vote (e.g. a referee
    # momentarily borrowing a player's track for one frame).
    by_track: Dict[int, List[FrameObservation]] = defaultdict(list)
    for o in targets:
        if o.track_id is not None:
            by_track[o.track_id].append(o)
    for tid, ol in by_track.items():
        ol.sort(key=lambda o: o.frame_idx)
        voted = [o.display_cls for o in ol]
        smoothed = list(voted)
        for i, o in enumerate(ol):
            lo = max(0, i - window)
            hi = min(len(ol), i + window + 1)
            wins: Dict[int, float] = defaultdict(float)
            for j in range(lo, hi):
                wins[voted[j]] += (ol[j].det_confidence or 0.1)
            smoothed[i] = max(wins, key=wins.get)
        for o, c in zip(ol, smoothed):
            o.display_cls = c

    for o in targets:
        if o.display_cls != o.cls_id:
            n_changed += 1

    logger.info(
        f"class voting: space-time tube (±{window}f, {radius_px}px) + temporal "
        f"majority -> relabelled {n_changed}/{len(targets)} detections"
    )
    return n_changed


# ---- short-gap interpolation (minimap continuity) -------------------------

# Bridge gaps up to this many frames where a track has no pitch position
# (calibration produced None, or the detector briefly lost the player). Keeps
# minimap markers from blinking in/out. ~0.6 s at 25 fps.
GAP_FILL_MAX_FRAMES = 16


def interpolate_track_gaps(
    state: ClipState, *, max_gap_frames: int = GAP_FILL_MAX_FRAMES
) -> int:
    """Fill short per-track pitch_xy gaps with linear interpolation.
    Synthesized observations are flagged ``synthetic=True`` so pass2 draws
    them only on the minimap, never as fabricated bboxes on the video.
    Returns the number of observations added."""
    by_track: Dict[int, List[FrameObservation]] = defaultdict(list)
    for obs in state.observations:
        if obs.track_id is None or obs.cls_id == 3:   # skip ball
            continue
        by_track[obs.track_id].append(obs)

    new_obs: List[FrameObservation] = []
    n_added = 0
    for tid, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.frame_idx)
        anchors = [o for o in obs_list if o.pitch_xy is not None]
        if len(anchors) < 2:
            continue
        present = {o.frame_idx for o in obs_list}
        for a, b in zip(anchors, anchors[1:]):
            gap = b.frame_idx - a.frame_idx
            if gap <= 1 or gap > max_gap_frames:
                continue
            ax, ay = a.pitch_xy
            bx, by_ = b.pitch_xy
            for f in range(a.frame_idx + 1, b.frame_idx):
                if f in present:
                    continue                          # real obs exists this frame
                t = (f - a.frame_idx) / gap
                px = ax + (bx - ax) * t
                py = ay + (by_ - ay) * t
                new_obs.append(
                    FrameObservation(
                        frame_idx=f,
                        bbox_xyxy=a.bbox_xyxy,
                        cls_id=a.cls_id,
                        det_confidence=0.0,
                        track_id=tid,
                        pitch_xy=(px, py),
                        team_id=a.team_id,
                        synthetic=True,
                        display_cls=a.display_cls if a.display_cls is not None else a.cls_id,
                    )
                )
                n_added += 1

    state.observations.extend(new_obs)
    logger.info(
        f"gap interpolation: added {n_added} synthetic minimap points "
        f"(max_gap={max_gap_frames}f)"
    )
    return n_added


def _ema_pass(
    obs_list: List[FrameObservation],
    *,
    alpha: float,
    gap_reset_frames: int,
    outlier_delta_m: float,
    outlier_alpha: float,
    dead_band_m: float,
    reverse: bool,
) -> int:
    """One EMA sweep over a track's observations (forward or reverse).

    Returns the number of outlier-damped steps.
    """
    seq = list(reversed(obs_list)) if reverse else obs_list
    n_out = 0
    last_frame = -10 ** 9 if not reverse else 10 ** 9
    sx: float = 0.0
    sy: float = 0.0
    initialized = False

    for obs in seq:
        x_raw, y_raw = obs.pitch_xy
        gap = abs(obs.frame_idx - last_frame)
        if not initialized or gap > gap_reset_frames:
            sx, sy = x_raw, y_raw
            initialized = True
        else:
            dx = x_raw - sx
            dy = y_raw - sy
            mag = (dx * dx + dy * dy) ** 0.5
            a = alpha
            if mag > outlier_delta_m * max(1, gap):
                a = outlier_alpha
                n_out += 1
            new_sx = a * x_raw + (1.0 - a) * sx
            new_sy = a * y_raw + (1.0 - a) * sy
            # dead-band: freeze if proposed delta is negligible
            if ((new_sx - sx) ** 2 + (new_sy - sy) ** 2) ** 0.5 > dead_band_m:
                sx, sy = new_sx, new_sy
        obs.pitch_xy = (sx, sy)
        last_frame = obs.frame_idx

    return n_out


def smooth_pitch_xy(
    state: ClipState,
    *,
    alpha: float = DEFAULT_ALPHA,
    gap_reset_frames: int = DEFAULT_GAP_RESET_FRAMES,
    outlier_delta_m: float = OUTLIER_DELTA_M,
    outlier_alpha: float = OUTLIER_ALPHA,
    dead_band_m: float = DEAD_BAND_M,
    two_pass: bool = True,
) -> int:
    """Smooth ``obs.pitch_xy`` per-track in-place: forward+backward EMA
    (symmetric low-pass, no phase lag), gap-reset after long absences,
    outlier damping for single-frame calibration glitches, and a dead-band
    that freezes near-static targets to kill micro-jitter."""
    if not state.observations:
        return 0

    by_track: Dict[int, List[FrameObservation]] = defaultdict(list)
    for obs in state.observations:
        if obs.track_id is None or obs.pitch_xy is None:
            continue
        if obs.cls_id == 3:        # ball: handled by filter_ball_outliers, not EMA
            continue
        by_track[obs.track_id].append(obs)

    n_smoothed = 0
    n_outliers = 0
    for tid, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.frame_idx)
        n_outliers += _ema_pass(
            obs_list,
            alpha=alpha, gap_reset_frames=gap_reset_frames,
            outlier_delta_m=outlier_delta_m, outlier_alpha=outlier_alpha,
            dead_band_m=dead_band_m, reverse=False,
        )
        if two_pass:
            _ema_pass(
                obs_list,
                alpha=alpha, gap_reset_frames=gap_reset_frames,
                outlier_delta_m=outlier_delta_m, outlier_alpha=outlier_alpha,
                dead_band_m=dead_band_m, reverse=True,
            )
        n_smoothed += len(obs_list)

    logger.info(
        f"track_smoother: pitch_xy EMA(alpha={alpha}, gap_reset={gap_reset_frames}f, "
        f"outlier>{outlier_delta_m}m@{outlier_alpha}, dead_band={dead_band_m}m, "
        f"two_pass={two_pass}) applied to {n_smoothed} obs / {len(by_track)} tracks, "
        f"{n_outliers} outliers damped"
    )
    return n_smoothed
