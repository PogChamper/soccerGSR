"""Multi-object tracker wrapper around our vendored BoT-SORT.

Per-clip instance — never shared across requests because tracker state
(Kalman filters, lost stracks, frame_count) is clip-scoped.

The vendored ``app.vendor.boxmot`` is a torch-free fork of BoT-SORT (see
``app/vendor/boxmot/__init__.py``). ReID is not built-in: pre-computed
appearance embeddings (DINOv3, see ``app.services.embedder``) are passed via
``embs`` to ``update()``. By default ``with_reid=True``; the job worker
falls back to ``with_reid=False`` (motion + IoU + CMC only) when the
embedder model is unavailable.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from app.services.detector import Detection

logger = logging.getLogger(__name__)


class BoxmotTracker:
    """BoT-SORT motion tracker (vendored, torch-free). One instance per clip."""

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        # max_time_lost = frame_rate/30 * track_buffer. At 25 fps a buffer of
        # 90 keeps a lost track alive ~3 s, long enough to re-acquire a player
        # who left frame during a camera pan via DINOv3 ReID instead of
        # spawning a fresh id (cuts fragmentation ~2-3x).
        track_buffer: int = 90,
        match_thresh: float = 0.8,
        cmc_method: str = "ecc",
        with_reid: bool = True,
        appearance_thresh: float = 0.4,
        proximity_thresh: float = 0.5,
        frame_rate: int = 30,
    ):
        from app.vendor.boxmot import BotSort

        logger.info(
            f"tracker: BotSort (vendored, no torch) cmc={cmc_method} "
            f"with_reid={with_reid} frame_rate={frame_rate}"
        )

        self._impl = BotSort(
            reid_weights=None,
            device=None,
            half=False,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            cmc_method=cmc_method,
            frame_rate=frame_rate,
            with_reid=with_reid,
        )
        self._with_reid = with_reid

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> List[Tuple[Detection, Optional[int]]]:
        """Update tracker, return list of (Detection, track_id) in **input order**.

        Detections that did not match any track receive ``track_id=None``.

        ``embeddings`` (optional, shape ``(N, D)``): pre-computed appearance
        features per detection, computed externally (e.g. by an ONNX ReID
        extractor). Required when ``with_reid=True``.
        """
        if not detections:
            self._impl.update(np.empty((0, 6), dtype=np.float32), frame)
            return []

        dets_np = np.array(
            [
                [
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    det.confidence,
                    det.class_id,
                ]
                for det in detections
            ],
            dtype=np.float32,
        )

        outputs = self._impl.update(dets_np, frame, embs=embeddings)
        # outputs shape: (M, 8) -> [x1, y1, x2, y2, track_id, conf, cls, det_ind]

        track_ids: List[Optional[int]] = [None] * len(detections)
        for row in outputs:
            det_ind = int(row[7])
            if 0 <= det_ind < len(track_ids):
                track_ids[det_ind] = int(row[4])

        return list(zip(detections, track_ids))


def make_tracker(frame_rate: int = 30, *, with_reid: bool = True) -> BoxmotTracker:
    """Factory used by ``VideoProcessor.tracker_factory``."""
    return BoxmotTracker(frame_rate=frame_rate, with_reid=with_reid)
