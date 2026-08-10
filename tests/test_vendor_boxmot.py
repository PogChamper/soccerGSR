import logging

import numpy as np
import pytest

from app.vendor.boxmot import BotSort
from app.vendor.boxmot.motion.cmc import get_cmc_method
from app.vendor.boxmot.motion.cmc.ecc import ECC
from app.vendor.boxmot.trackers.basetracker import BaseTracker
from app.vendor.boxmot.utils import logger


def test_cmc_surface_is_limited_to_ecc() -> None:
    assert get_cmc_method(" ECC ") is ECC
    assert get_cmc_method(None) is None

    with pytest.raises(ValueError, match="Supported values: ecc"):
        get_cmc_method("orb")


def test_vendor_uses_application_logging_without_handlers() -> None:
    assert isinstance(logger, logging.Logger)
    assert logger.handlers == []
    assert logger.propagate is True


def test_base_tracker_has_no_rendering_surface() -> None:
    assert not hasattr(BaseTracker, "plot_results")
    assert not hasattr(BaseTracker, "iter_tracks_for_display")


def test_botsort_accepts_an_empty_frame_with_ecc() -> None:
    tracker = BotSort(with_reid=False, cmc_method="ecc")
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    output = tracker.update(np.empty((0, 6), dtype=np.float32), frame)

    assert output.shape == (0, 8)


def test_reid_accepts_an_empty_frame() -> None:
    tracker = BotSort(with_reid=True, cmc_method="ecc")
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    output = tracker.update(
        np.empty((0, 6), dtype=np.float32),
        frame,
        embs=np.empty((0, 512), dtype=np.float32),
    )

    assert output.shape == (0, 8)


def test_botsort_keeps_identity_across_frames() -> None:
    tracker = BotSort(with_reid=False, cmc_method="ecc")
    y, x = np.indices((96, 128))
    gray = (((x // 8) + (y // 8)) % 2 * 255).astype(np.uint8)
    frame = np.repeat(gray[..., None], 3, axis=2)
    detections = np.asarray([[20, 10, 40, 70, 0.95, 0]], dtype=np.float32)

    first = tracker.update(detections, frame)
    second = tracker.update(detections, frame)

    assert first.shape == second.shape == (1, 8)
    assert first[0, 4] == second[0, 4]
