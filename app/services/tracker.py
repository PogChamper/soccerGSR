"""Per-clip BoT-SORT wrapper using external OSNet embeddings."""

from __future__ import annotations

import logging

import numpy as np

from app.services.detector import Detection

logger = logging.getLogger(__name__)
_EMBEDDING_DIM = 512
_HUMAN_GROUP = 0
_BALL_GROUP = 1


class BoxmotTracker:
    """BoT-SORT motion tracker (vendored, torch-free). One instance per clip."""

    def __init__(
        self,
        *,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 90,
        match_thresh: float = 0.8,
        cmc_method: str = "ecc",
        max_cosine_distance: float = 0.4,
        proximity_thresh: float = 0.5,
        frame_rate: int = 30,
    ):
        from app.vendor.boxmot import BotSort

        logger.info(
            "tracker: BoT-SORT cmc=%s frame_rate=%d max_cosine_distance=%.2f",
            cmc_method,
            frame_rate,
            max_cosine_distance,
        )

        options = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": max_cosine_distance,
            "cmc_method": cmc_method,
            "frame_rate": frame_rate,
            "with_reid": True,
        }
        self._human_impl = BotSort(**options)
        self._ball_impl = BotSort(**options)
        self._track_ids: dict[tuple[int, int], int] = {}
        self._next_track_id = 1

    def _assign_track_id(self, group: int, internal_id: int) -> int:
        key = (group, internal_id)
        track_id = self._track_ids.get(key)
        if track_id is None:
            track_id = self._next_track_id
            self._track_ids[key] = track_id
            self._next_track_id += 1
        return track_id

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> list[tuple[Detection, int | None]]:
        """Return detections and assigned track IDs in input order.

        Detections that did not match any track receive ``track_id=None``.

        ``embeddings`` must contain one external OSNet feature per detection.
        """
        if embeddings is None:
            if detections:
                raise ValueError("OSNet embeddings are required for non-empty detections")
            embeddings = np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
        if embeddings.shape != (len(detections), _EMBEDDING_DIM):
            raise ValueError(f"embeddings must have shape ({len(detections)}, {_EMBEDDING_DIM})")

        dets_np = np.asarray(
            [
                [
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[2],
                    det.bbox[3],
                    det.confidence,
                    _BALL_GROUP if det.class_id == 3 else _HUMAN_GROUP,
                ]
                for det in detections
            ],
            dtype=np.float32,
        ).reshape(-1, 6)

        track_ids: list[int | None] = [None] * len(detections)
        groups = (
            (_HUMAN_GROUP, self._human_impl),
            (_BALL_GROUP, self._ball_impl),
        )
        for group, implementation in groups:
            indices = np.flatnonzero(dets_np[:, 5] == group)
            outputs = implementation.update(
                dets_np[indices],
                frame,
                embs=embeddings[indices],
            )
            # [x1, y1, x2, y2, track_id, confidence, class_id, detection_index]
            for row in outputs:
                local_index = int(row[7])
                if 0 <= local_index < len(indices):
                    input_index = int(indices[local_index])
                    track_ids[input_index] = self._assign_track_id(group, int(row[4]))

        return list(zip(detections, track_ids))


def make_tracker(frame_rate: int = 30) -> BoxmotTracker:
    """Factory used by ``VideoProcessor.tracker_factory``."""
    return BoxmotTracker(frame_rate=frame_rate)
