from __future__ import annotations

import numpy as np

from app.services.keypoints import HRNetKeypointsExtractor
from app.vendor.pnlcalib.utils import utils_heatmap


class _Session:
    def run(self, output_names, inputs):
        assert output_names is None
        assert next(iter(inputs.values())).shape == (1, 3, 540, 960)
        return [np.ones((1, 3, 4, 4), dtype=np.float32)]


def test_extract_uses_validated_ground_plane_decoder(monkeypatch) -> None:
    calls: list[tuple[float, bool]] = []
    keypoint_coordinates = object()
    line_coordinates = object()

    monkeypatch.setattr(
        utils_heatmap,
        "get_keypoints_from_heatmap_batch_maxpool",
        lambda heatmaps: keypoint_coordinates,
    )
    monkeypatch.setattr(
        utils_heatmap,
        "get_keypoints_from_heatmap_batch_maxpool_l",
        lambda heatmaps: line_coordinates,
    )

    def coordinates_to_dict(coordinates, *, threshold, ground_plane_only):
        calls.append((threshold, ground_plane_only))
        label = 1 if coordinates is keypoint_coordinates else 2
        return [{label: (0.1, 0.2, 0.9)}]

    monkeypatch.setattr(utils_heatmap, "coords_to_dict", coordinates_to_dict)
    monkeypatch.setattr(
        utils_heatmap,
        "complete_keypoints",
        lambda keypoints, lines, **kwargs: (keypoints, lines),
    )

    extractor = HRNetKeypointsExtractor.__new__(HRNetKeypointsExtractor)
    extractor.kp_sess = _Session()
    extractor.lines_sess = _Session()
    extractor.kp_in_name = "images"
    extractor.lines_in_name = "images"

    keypoints, lines = extractor.extract(np.zeros((90, 160, 3), dtype=np.uint8))

    assert keypoints == {1: (0.1, 0.2, 0.9)}
    assert lines == {2: (0.1, 0.2, 0.9)}
    assert calls == [(0.1611, True), (0.3434, True)]
