from __future__ import annotations

import cv2
import numpy as np

from app.vendor.boxmot.utils import logger as LOGGER


class ECC:
    """
    OpenCV ECC-based motion estimation using cv2.findTransformECC.
    Produces:
      - 2x3 affine-like matrix for TRANSLATION/EUCLIDEAN/AFFINE
      - 3x3 homography matrix for HOMOGRAPHY
    """

    def __init__(
        self,
        warp_mode: int = cv2.MOTION_TRANSLATION,
        eps: float = 1e-5,
        max_iter: int = 100,
        scale: float = 0.15,
        grayscale: bool = True,
    ) -> None:
        self.grayscale = bool(grayscale)
        self.scale = float(scale)
        self.warp_mode = int(warp_mode)

        self.termination_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(max_iter),
            float(eps),
        )

        self.prev_img: np.ndarray | None = None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img
        return cv2.resize(out, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

    def apply(self, img: np.ndarray) -> np.ndarray:
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        if self.prev_img is None:
            self.prev_img = self._preprocess(img)
            return warp_matrix

        curr = self._preprocess(img)

        try:
            _, warp_matrix = cv2.findTransformECC(
                self.prev_img,
                curr,
                warp_matrix,
                self.warp_mode,
                self.termination_criteria,
                None,
                1,
            )
        except cv2.error as e:
            if e.code == cv2.Error.StsNoConv:
                LOGGER.warning("ECC did not converge; returning identity warp.")
                self.prev_img = curr
                return warp_matrix
            raise

        # upscale translation back to original image coordinates
        if self.scale < 1.0:
            warp_matrix = warp_matrix.copy()
            warp_matrix[0, 2] /= self.scale
            warp_matrix[1, 2] /= self.scale

        self.prev_img = curr
        return warp_matrix
