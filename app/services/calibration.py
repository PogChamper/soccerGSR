"""Pitch calibration from PnLCalib keypoints and line extremities."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.signal import savgol_filter

from app.services.clip_state import ClipState, FrameInfo

logger = logging.getLogger(__name__)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Reject solves whose reprojection error reported by the solver is clearly bad.
REP_ERR_MAX_PX = 20.0
# Reject a solve whose quad corners move more than this per frame since the
# last accepted one (budget widens over unsolved gaps, capped at H_JUMP_GAP_CAP
# frames), unless the jump persists (a real camera cut) - then accept it as
# the new reference.
H_JUMP_M = 5.0
H_JUMP_GAP_CAP = 4
H_JUMP_MAX_REJECT = 6
# Longest hold of a homography into an unsolved gap; gaps up to twice this
# are interpolated end to end, longer ones stay uncalibrated in the middle.
H_HOLD_FRAMES = 6
# Savitzky-Golay window on the quad trajectory (0.3 s at 30 fps).
H_SMOOTH_WINDOW = 9


def _image_quad(width: int, height: int) -> np.ndarray:
    """Fixed image quad in the band where players actually stand."""
    return np.array(
        [
            (0.20 * width, 0.55 * height),
            (0.80 * width, 0.55 * height),
            (0.80 * width, 0.95 * height),
            (0.20 * width, 0.95 * height),
        ]
    )


def _quad_world(homography: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Project the image quad to pitch metres; None if degenerate."""
    points = np.hstack([quad, np.ones((4, 1))]) @ homography.T
    if np.any(np.abs(points[:, 2]) <= 1e-9):
        return None
    world = points[:, :2] / points[:, 2:3]
    return world if np.isfinite(world).all() else None


def _signed_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


class PnLCalibrator:
    """Calibrate one clip and project observation feet to pitch coordinates.

    Per-frame homographies are gated, gap-filled and smoothed as trajectories
    of a fixed image quad projected to pitch metres: four point pairs determine
    a homography exactly, and metres are the units the minimap cares about.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        *,
        refine_lines: bool = False,
    ) -> None:
        from app.vendor.pnlcalib.utils.utils_calib import FramebyFrameCalib

        self._camera = FramebyFrameCalib(
            iwidth=frame_width,
            iheight=frame_height,
            denormalize=True,
        )
        self._refine_lines = refine_lines
        self._quad = _image_quad(frame_width, frame_height)

    def calibrate_one(self, keypoints: dict, lines: dict) -> np.ndarray | None:
        try:
            self._camera.update(keypoints, lines)
            result = self._camera.heuristic_voting_ground(refine_lines=self._refine_lines)
            if result is None or "homography" not in result:
                return None
            rep_err = result.get("rep_err")
            if rep_err is not None and rep_err > REP_ERR_MAX_PX:
                return None
            homography = np.asarray(result["homography"], dtype=np.float64)
            if homography.shape != (3, 3) or not np.isfinite(homography).all():
                return None
            return homography
        except Exception as exc:
            logger.debug("calibration failed: %s", exc)
            return None

    @staticmethod
    def _store(
        frame: FrameInfo,
        homography_image_to_world: np.ndarray,
    ) -> None:
        homography_world_to_image = np.linalg.inv(homography_image_to_world)
        frame.homography_image_to_world = homography_image_to_world.tolist()
        frame.homography_world_to_image = homography_world_to_image.tolist()

    @staticmethod
    def _gate(quads: list[np.ndarray | None]) -> tuple[int, set[int]]:
        """Drop degenerate and jumping quads in place.

        Returns the kept count and the indices where a persistent jump was
        accepted as a camera cut (reseeds).
        """
        areas = [_signed_area(quad) for quad in quads if quad is not None]
        if not areas:
            return 0, set()
        sign = 1.0 if float(np.median(areas)) >= 0 else -1.0
        reference = float(np.median(np.abs(areas)))

        last: np.ndarray | None = None
        last_index = 0
        rejects = 0
        kept = 0
        reseeds: set[int] = set()
        for index, quad in enumerate(quads):
            if quad is None:
                continue
            area = _signed_area(quad) * sign
            if not reference / 4 <= area <= reference * 4:
                quads[index] = None
                continue
            if last is not None:
                budget = H_JUMP_M * min(index - last_index, H_JUMP_GAP_CAP)
                if np.linalg.norm(quad - last, axis=1).max() > budget:
                    rejects += 1
                    if rejects <= H_JUMP_MAX_REJECT:
                        quads[index] = None
                        continue
                    reseeds.add(index)
            rejects = 0
            last = quad
            last_index = index
            kept += 1
        return kept, reseeds

    @staticmethod
    def _fill(quads: list[np.ndarray | None], reseeds: set[int]) -> list[str]:
        """Interpolate short gaps, hold the edges of long ones.

        A gap ending in a reseed is a camera cut: never interpolated, and at
        least one frame at the junction stays "none" so smoothing and the
        renderer treat the two sides as separate runs.
        """
        n = len(quads)
        sources = ["solved" if quad is not None else "none" for quad in quads]
        solved = [index for index in range(n) if quads[index] is not None]
        if not solved:
            return sources

        def hold(source: int, targets: range) -> None:
            for index in targets:
                quads[index] = quads[source]
                sources[index] = "held"

        hold(solved[0], range(max(0, solved[0] - H_HOLD_FRAMES), solved[0]))
        hold(solved[-1], range(solved[-1] + 1, min(n, solved[-1] + 1 + H_HOLD_FRAMES)))
        for left, right in zip(solved, solved[1:]):
            gap = right - left - 1
            if gap == 0:
                continue
            if gap <= 2 * H_HOLD_FRAMES and right not in reseeds:
                for index in range(left + 1, right):
                    fraction = (index - left) / (right - left)
                    quads[index] = (1 - fraction) * quads[left] + fraction * quads[right]
                    sources[index] = "interp"
            else:
                left_end = min(left + 1 + H_HOLD_FRAMES, left + 1 + gap // 2)
                right_start = max(right - H_HOLD_FRAMES, left_end + 1)
                hold(left, range(left + 1, left_end))
                hold(right, range(right_start, right))
        return sources

    @staticmethod
    def _smooth(quads: list[np.ndarray | None]) -> None:
        """Savitzky-Golay over each contiguous run of quads."""
        n = len(quads)
        start = 0
        while start < n:
            if quads[start] is None:
                start += 1
                continue
            end = start
            while end < n and quads[end] is not None:
                end += 1
            run = end - start
            if run >= 5:
                window = min(H_SMOOTH_WINDOW, run if run % 2 else run - 1)
                flat = np.stack(quads[start:end]).reshape(run, 8)
                flat = savgol_filter(flat, window, 2, axis=0, mode="interp")
                quads[start:end] = list(flat.reshape(run, 4, 2))
            start = end

    def calibrate(self, state: ClipState) -> None:
        frames = sorted(state.frames, key=lambda frame: frame.frame_idx)
        quads: list[np.ndarray | None] = []
        for frame in frames:
            homography = self.calibrate_one(frame.keypoints, frame.lines)
            quads.append(None if homography is None else _quad_world(homography, self._quad))

        solved, reseeds = self._gate(quads)
        sources = self._fill(quads, reseeds)
        self._smooth(quads)

        stored = 0
        for frame, quad, source in zip(frames, quads, sources):
            frame.homography_source = source
            if quad is None:
                continue
            homography, _ = cv2.findHomography(self._quad, quad)
            if (
                homography is None
                or not np.isfinite(homography).all()
                or abs(np.linalg.det(homography)) < 1e-12
            ):
                frame.homography_source = "none"
                continue
            self._store(frame, homography)
            stored += 1

        frame_lookup = {frame.frame_idx: frame for frame in frames}
        projected = 0
        for observation in state.observations:
            frame = frame_lookup.get(observation.frame_idx)
            if (
                frame is None
                or frame.homography_image_to_world is None
                or observation.foot_xy_image is None
            ):
                continue
            homography = np.asarray(frame.homography_image_to_world, dtype=np.float64)
            point = homography @ np.asarray((*observation.foot_xy_image, 1.0), dtype=np.float64)
            if abs(point[2]) <= 1e-9:
                continue
            pitch_x, pitch_y = point[:2] / point[2]
            if not (-PITCH_LENGTH_M / 2 - 5 <= pitch_x <= PITCH_LENGTH_M / 2 + 5):
                continue
            if not (-PITCH_WIDTH_M / 2 - 5 <= pitch_y <= PITCH_WIDTH_M / 2 + 5):
                continue
            observation.pitch_xy = float(pitch_x), float(pitch_y)
            projected += 1

        logger.info(
            "calibration: solved=%d stored=%d projected=%d/%d",
            solved,
            stored,
            projected,
            len(state.observations),
        )


def make_calibrator(frame_width: int, frame_height: int) -> PnLCalibrator:
    return PnLCalibrator(frame_width, frame_height)
