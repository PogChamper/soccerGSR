"""Per-frame camera calibration via PnLCalib FramebyFrameCalib: keypoints +
lines -> cam_params -> ground-plane homographies (with temporal
stabilisation) -> observation foot points projected to pitch metres."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from app.services.clip_state import ClipState

logger = logging.getLogger(__name__)


# Field is 105 x 68 m, origin in PnLCalib is at center (-52.5..52.5, -34..34)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Camera-parameter EMA. Disabled (1.0) by default: heuristic_voting can flip
# between two valid candidate calibrations on adjacent frames, and blending
# incompatible cam_params yields a geometrically wrong camera that increases
# jitter. Lower only if cam_params are verified stable on the clip.
CAM_EMA_ALPHA = 1.0
CAM_EMA_GAP_RESET = 12

# Ground-homography stabilisation (the actual anti-"breathing" mechanism):
# EMA-smooth H across frames; if a new H moves the reprojected pitch corners
# more than H_SWITCH_PX away, treat it as a candidate switch and keep the
# smoothed H — unless it persists H_SWITCH_MAX_REJECT frames (camera really
# moved -> re-seed). On outright calibration failure hold the last good H up
# to H_HOLD_FRAMES so markers don't blink.
H_EMA_ALPHA = 0.35
H_SWITCH_PX = 60.0
H_SWITCH_MAX_REJECT = 6
H_HOLD_FRAMES = 3


_PITCH_CORNERS_W = np.array(
    [
        [-PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2, 1.0],
        [PITCH_LENGTH_M / 2, -PITCH_WIDTH_M / 2, 1.0],
        [PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2, 1.0],
        [-PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2, 1.0],
    ],
    dtype=np.float64,
)


def _corner_reproj_diff(Ha: np.ndarray, Hb: np.ndarray) -> float:
    """Mean pixel distance between the pitch corners as projected by two
    world->image homographies. A scale-free way to compare two H matrices in
    the units that actually matter (screen pixels)."""
    pa = (Ha @ _PITCH_CORNERS_W.T).T
    pb = (Hb @ _PITCH_CORNERS_W.T).T
    wa = pa[:, 2:3]
    wb = pb[:, 2:3]
    if np.any(np.abs(wa) < 1e-9) or np.any(np.abs(wb) < 1e-9):
        return float("inf")
    pa = pa[:, :2] / wa
    pb = pb[:, :2] / wb
    return float(np.linalg.norm(pa - pb, axis=1).mean())


def _ema_homography(H_prev: np.ndarray, H_new: np.ndarray, alpha: float) -> np.ndarray:
    """Element-wise EMA of two homographies, renormalised so H[2,2] == 1."""
    H = alpha * H_new + (1.0 - alpha) * H_prev
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Snap a (nearly-orthogonal) 3x3 matrix back to the closest SO(3)
    rotation via SVD. Needed after EMA-blending two rotation matrices —
    component-wise blending does not preserve orthogonality.
    """
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:           # ensure right-handed
        U[:, -1] *= -1
        R_orth = U @ Vt
    return R_orth


def _ema_cam_params(prev: dict, new: dict, alpha: float) -> dict:
    """Blend two cam_params dicts: linear EMA on scalars / vectors,
    component-wise EMA + SO(3) re-projection on rotation matrix."""
    def _blend_arr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return alpha * b + (1.0 - alpha) * a

    out = {}
    for k in new.keys():
        if k == "rotation_matrix":
            R_prev = np.asarray(prev[k], dtype=np.float64)
            R_new = np.asarray(new[k], dtype=np.float64)
            R_blend = alpha * R_new + (1.0 - alpha) * R_prev
            out[k] = _project_to_so3(R_blend)
        elif k in (
            "x_focal_length", "y_focal_length",
        ):
            out[k] = float(_blend_arr(prev[k], new[k]))
        elif k in ("principal_point", "position_meters"):
            out[k] = _blend_arr(prev[k], new[k]).tolist()
        else:
            out[k] = new[k]
    return out


def _projection_from_cam_params(cam_params: dict) -> np.ndarray:
    x_focal = cam_params["x_focal_length"]
    y_focal = cam_params["y_focal_length"]
    pp = np.array(cam_params["principal_point"], dtype=np.float64)
    pos = np.array(cam_params["position_meters"], dtype=np.float64)
    rot = np.array(cam_params["rotation_matrix"], dtype=np.float64)

    It = np.eye(4)[:-1]
    It[:, -1] = -pos
    K = np.array([[x_focal, 0, pp[0]], [0, y_focal, pp[1]], [0, 0, 1]], dtype=np.float64)
    P = K @ (rot @ It)  # (3, 4)
    return P


def _ground_homography_from_P(P: np.ndarray) -> np.ndarray:
    """For points on z=0 plane: image = P @ [X, Y, 0, 1]^T.

    -> H_world2img = [P[:,0], P[:,1], P[:,3]] (3x3).
    """
    return np.stack([P[:, 0], P[:, 1], P[:, 3]], axis=1)


class PnLCalibrator:
    """Per-clip calibrator. PnLCalib's FramebyFrameCalib is stateless across frames
    (each .update overwrites internal kp/lines), so one instance per clip is fine.
    """

    def __init__(self, frame_w: int, frame_h: int, *, refine_lines: bool = False):
        from app.vendor.pnlcalib.utils.utils_calib import FramebyFrameCalib

        # PnLCalib's iwidth/iheight is the ORIGINAL frame size; with denormalize=True
        # it un-scales the [0,1] kp coordinates back to original frame pixels.
        self._cam = FramebyFrameCalib(iwidth=frame_w, iheight=frame_h, denormalize=True)
        self.refine_lines = refine_lines
        self.frame_w = frame_w
        self.frame_h = frame_h

    def calibrate_one(
        self,
        kp_dict: dict,
        lines_dict: dict,
    ) -> Optional[dict]:
        """Returns {cam_params, P, H_w2i, H_i2w} or None if calibration failed."""
        try:
            self._cam.update(kp_dict, lines_dict)
            res = self._cam.heuristic_voting(refine_lines=self.refine_lines)
        except Exception as exc:
            logger.debug(f"calibrate_one: {exc}")
            return None
        if res is None or "cam_params" not in res:
            return None

        cam_params = res["cam_params"]
        try:
            P = _projection_from_cam_params(cam_params)
            H_w2i = _ground_homography_from_P(P)
            H_i2w = np.linalg.inv(H_w2i)
        except Exception as exc:
            logger.debug(f"projection failed: {exc}")
            return None

        return {
            "cam_params": cam_params,
            "P": P,
            "H_w2i": H_w2i,
            "H_i2w": H_i2w,
        }

    def _store_H(self, frame, H_w2i: np.ndarray, cam_params: Optional[dict] = None) -> None:
        """Write a world->image homography (and its inverse) onto a frame."""
        try:
            H_i2w = np.linalg.inv(H_w2i)
        except np.linalg.LinAlgError:
            return
        if cam_params is not None:
            frame.cam_params = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in cam_params.items()
            }
        frame.homography_world_to_image = (
            H_w2i.tolist() if hasattr(H_w2i, "tolist") else H_w2i
        )
        frame.homography_image_to_world = H_i2w.tolist()

    def calibrate(self, state: ClipState) -> None:
        """Iterate frames in state, fill calibration fields, then project foot
        points of observations into pitch coords.

        Camera parameters are smoothed across frames with an EMA on
        cam_params (rotation matrix re-projected to SO(3) after blending).
        Resets if a long gap with no successful calibration appears.
        """
        n_calibrated = 0          # frames with a real (non-rejected) solution
        n_held = 0                # frames filled by holding the smoothed H
        n_rejected = 0            # candidate-switch frames replaced by smoothed H
        H_ema: Optional[np.ndarray] = None
        last_good_frame: int = -10 ** 9
        reject_run = 0

        for frame in sorted(state.frames, key=lambda f: f.frame_idx):
            res = self.calibrate_one(frame.keypoints, frame.lines)
            gap = frame.frame_idx - last_good_frame

            if res is None:
                # Calibration failed entirely. Hold the last smoothed H for a
                # short window so markers don't blink on brief dropouts.
                if H_ema is not None and 0 < gap <= H_HOLD_FRAMES:
                    self._store_H(frame, H_ema)
                    n_held += 1
                continue

            H_raw = res["H_w2i"]
            if H_ema is None or gap > CAM_EMA_GAP_RESET:
                # cold start / long gap -> trust raw, reset filter
                H_use = H_raw
                H_ema = H_raw
                reject_run = 0
            else:
                diff = _corner_reproj_diff(H_raw, H_ema)
                if diff <= H_SWITCH_PX:
                    H_ema = _ema_homography(H_ema, H_raw, H_EMA_ALPHA)
                    H_use = H_ema
                    reject_run = 0
                else:
                    # candidate switch / outlier
                    reject_run += 1
                    if reject_run >= H_SWITCH_MAX_REJECT:
                        H_use = H_raw          # camera genuinely moved -> re-seed
                        H_ema = H_raw
                        reject_run = 0
                    else:
                        H_use = H_ema          # keep stable estimate, drop raw
                        n_rejected += 1

            self._store_H(frame, H_use, cam_params=res.get("cam_params"))
            n_calibrated += 1
            last_good_frame = frame.frame_idx

        # Project foot points to pitch
        n_projected = 0
        h_by_frame = {f.frame_idx: f for f in state.frames}
        for obs in state.observations:
            f = h_by_frame.get(obs.frame_idx)
            if f is None or f.homography_image_to_world is None or obs.foot_xy_image is None:
                continue
            H = np.array(f.homography_image_to_world, dtype=np.float64)
            pt = np.array([obs.foot_xy_image[0], obs.foot_xy_image[1], 1.0], dtype=np.float64)
            wpt = H @ pt
            if abs(wpt[2]) < 1e-9:
                continue
            wpt /= wpt[2]
            x_m, y_m = float(wpt[0]), float(wpt[1])
            # Centered coords; clip to ~field bounds with margin
            if not (-PITCH_LENGTH_M / 2 - 5 <= x_m <= PITCH_LENGTH_M / 2 + 5):
                continue
            if not (-PITCH_WIDTH_M / 2 - 5 <= y_m <= PITCH_WIDTH_M / 2 + 5):
                continue
            obs.pitch_xy = (x_m, y_m)
            n_projected += 1

        logger.info(
            f"calibration: {n_calibrated}/{len(state.frames)} frames calibrated "
            f"(+{n_held} held, {n_rejected} candidate-switches rejected), "
            f"{n_projected}/{len(state.observations)} observations projected to pitch"
        )


def make_calibrator(frame_w: int, frame_h: int) -> PnLCalibrator:
    return PnLCalibrator(frame_w, frame_h)
