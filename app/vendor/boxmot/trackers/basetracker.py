from abc import ABC, abstractmethod

import numpy as np

DET_COLS = 6  # x1, y1, x2, y2, conf, cls
OUTPUT_COLS = 8  # x1, y1, x2, y2, id, conf, cls, det_ind


class BaseTracker(ABC):
    def __init__(self):
        self.frame_count = 0
        self.active_tracks = []

    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """Return one row per confirmed track: x1,y1,x2,y2,id,conf,cls,det_ind."""

    def empty_output(self, dtype=float) -> np.ndarray:
        return np.empty((0, OUTPUT_COLS), dtype=dtype)

    def check_inputs(self, dets, embs=None):
        assert dets.shape[1] == DET_COLS, (
            f"dets must have {DET_COLS} columns (x1,y1,x2,y2,conf,cls)"
        )
        if embs is not None:
            assert dets.shape[0] == embs.shape[0], "detections and embeddings sizes differ"
