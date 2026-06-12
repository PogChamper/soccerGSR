"""Field keypoints + line extremities extraction via PnLCalib HRNet ONNX.

Each frame:
1. Resize BGR -> 540x960 RGB
2. Run kp HRNet -> heatmaps (1, 58, h_h, w_h) — last channel is bg
3. Run lines HRNet -> heatmaps (1, 25, h_h, w_h)
4. Decode via PnLCalib heatmap utils -> kp_dict, lines_dict (normalized 0..1)

Input image size for HRNet is fixed (540x960). Output dicts are passed to
``app.services.calibration.PnLCalibrator``.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

from app.utils.cuda_env import get_providers
from app.utils.models_registry import ensure_model

logger = logging.getLogger(__name__)

HRNET_W, HRNET_H = 960, 540
KP_THRESHOLD = 0.3434      # PnLCalib defaults
LINE_THRESHOLD = 0.7867


class HRNetKeypointsExtractor:
    """ONNX-only inference of PnLCalib keypoint + line HRNets."""

    def __init__(self):
        import onnxruntime as ort

        kp_path = ensure_model("hrnet_kp")
        lines_path = ensure_model("hrnet_lines")
        providers = get_providers(prefer_gpu=True)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.kp_sess = ort.InferenceSession(str(kp_path), sess_opts, providers=providers)
        self.lines_sess = ort.InferenceSession(str(lines_path), sess_opts, providers=providers)
        self.kp_in_name = self.kp_sess.get_inputs()[0].name
        self.lines_in_name = self.lines_sess.get_inputs()[0].name
        logger.info(
            f"keypoints: kp={kp_path.name} lines={lines_path.name} "
            f"providers={self.kp_sess.get_providers()}"
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != HRNET_W or rgb.shape[0] != HRNET_H:
            rgb = cv2.resize(rgb, (HRNET_W, HRNET_H), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0  # PnLCalib uses to_tensor (0..1) without ImageNet norm
        arr = arr.transpose(2, 0, 1)
        return arr[None, ...]

    def extract(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Tuple[float, float, float]]]:
        """Return (kp_dict, lines_dict) where each is {label_int: (x_norm, y_norm, p)}.

        Coordinates are normalized to [0,1] relative to HRNet input (so
        downstream calibration expects ``denormalize=True`` in
        ``FramebyFrameCalib`` to scale to actual frame size).

        Pure numpy/scipy decode — no torch involved at runtime.
        """
        from app.vendor.pnlcalib.utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        inp = self._preprocess(frame_bgr)
        heatmaps_kp = self.kp_sess.run(None, {self.kp_in_name: inp})[0]
        heatmaps_lines = self.lines_sess.run(None, {self.lines_in_name: inp})[0]

        # drop the trailing background channel before peak extraction
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_lines[:, :-1, :, :])

        kp_dict = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
        lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=HRNET_W, h=HRNET_H, normalize=True
        )

        return kp_dict, lines_dict


_INSTANCE = None


def get_keypoints_extractor() -> HRNetKeypointsExtractor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HRNetKeypointsExtractor()
    return _INSTANCE
