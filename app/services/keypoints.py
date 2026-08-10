"""PnLCalib HRNet inference for field keypoints and line extremities."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.config import get_settings
from app.utils.cuda_env import get_providers
from app.utils.models_registry import ensure_model

logger = logging.getLogger(__name__)
settings = get_settings()

HRNET_WIDTH = 960
HRNET_HEIGHT = 540
KEYPOINT_THRESHOLD = 0.1611
LINE_THRESHOLD = 0.3434


class HRNetKeypointsExtractor:
    """ONNX-only inference of PnLCalib keypoint + line HRNets."""

    def __init__(self) -> None:
        import onnxruntime as ort

        kp_path = ensure_model("hrnet_kp", auto_download=settings.model_auto_download)
        lines_path = ensure_model("hrnet_lines", auto_download=settings.model_auto_download)
        providers = get_providers(prefer_gpu=True)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.kp_sess = ort.InferenceSession(str(kp_path), sess_opts, providers=providers)
        self.lines_sess = ort.InferenceSession(str(lines_path), sess_opts, providers=providers)
        self.kp_in_name = self.kp_sess.get_inputs()[0].name
        self.lines_in_name = self.lines_sess.get_inputs()[0].name
        logger.info(
            "HRNet loaded: keypoints=%s lines=%s providers=%s",
            kp_path.name,
            lines_path.name,
            self.kp_sess.get_providers(),
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != HRNET_WIDTH or rgb.shape[0] != HRNET_HEIGHT:
            rgb = cv2.resize(
                rgb,
                (HRNET_WIDTH, HRNET_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        tensor = rgb.astype(np.float32) / 255.0
        return tensor.transpose(2, 0, 1)[None]

    def extract(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
        """Return normalized ground-plane keypoints and line extremities."""
        from app.vendor.pnlcalib.utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        inp = self._preprocess(frame_bgr)
        heatmaps_kp = self.kp_sess.run(None, {self.kp_in_name: inp})[0]
        heatmaps_lines = self.lines_sess.run(None, {self.lines_in_name: inp})[0]

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_lines[:, :-1, :, :])

        kp_dict = coords_to_dict(
            kp_coords,
            threshold=KEYPOINT_THRESHOLD,
            ground_plane_only=True,
        )
        lines_dict = coords_to_dict(
            line_coords,
            threshold=LINE_THRESHOLD,
            ground_plane_only=True,
        )
        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0],
            lines_dict[0],
            w=HRNET_WIDTH,
            h=HRNET_HEIGHT,
            normalize=True,
        )

        return kp_dict, lines_dict


_INSTANCE: HRNetKeypointsExtractor | None = None


def get_keypoints_extractor() -> HRNetKeypointsExtractor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HRNetKeypointsExtractor()
    return _INSTANCE
