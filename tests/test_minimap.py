import numpy as np
import pytest

from app.services.minimap import MinimapRenderer


@pytest.mark.parametrize("shape", [(240, 320), (300, 500), (1, 1)])
def test_overlay_scales_to_small_frames(shape: tuple[int, int]) -> None:
    height, width = shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    renderer = MinimapRenderer()

    result = renderer.overlay(frame, [], {})

    assert result is frame
