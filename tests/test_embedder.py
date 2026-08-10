from __future__ import annotations

import numpy as np
import pytest

from app.services.embedder import OSNetEmbedder


class _Session:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def run(self, output_names, inputs):
        assert output_names == ["embeddings"]
        batch = inputs["images"]
        self.batch_sizes.append(len(batch))
        output = np.zeros((len(batch), 512), dtype=np.float32)
        output[:, 0] = 3.0
        output[:, 1] = 4.0
        return [output]


def _embedder(batch_size: int = 2) -> tuple[OSNetEmbedder, _Session]:
    session = _Session()
    embedder = OSNetEmbedder.__new__(OSNetEmbedder)
    embedder._session = session
    embedder._input_name = "images"
    embedder._output_name = "embeddings"
    embedder._batch_size = batch_size
    return embedder, session


def test_preprocess_matches_osnet_input_contract() -> None:
    red_bgr = np.zeros((32, 16, 3), dtype=np.uint8)
    red_bgr[:, :, 2] = 255

    batch = OSNetEmbedder._preprocess([red_bgr])

    assert batch.shape == (1, 3, 256, 128)
    assert batch.dtype == np.float32
    np.testing.assert_allclose(
        batch[0, :, 0, 0], [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    )


def test_embed_crops_batches_and_normalizes_features() -> None:
    embedder, session = _embedder(batch_size=2)
    crops = [np.zeros((16, 8, 3), dtype=np.uint8) for _ in range(3)]

    embeddings = embedder.embed_crops(crops)

    assert embeddings.shape == (3, 512)
    assert session.batch_sizes == [2, 1]
    np.testing.assert_allclose(embeddings[:, :2], [[0.6, 0.8]] * 3)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_embed_boxes_embeds_degenerate_boxes() -> None:
    embedder, _ = _embedder()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    embeddings = embedder.embed_boxes(frame, [(0, 0, 10, 10), (4, 4, 5, 5), (10, 10, 20, 20)])

    np.testing.assert_allclose(embeddings[:, :2], [[0.6, 0.8]] * 3)


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_embed_crops_rejects_invalid_model_output(value: float) -> None:
    embedder, session = _embedder()

    def run(_output_names, inputs):
        return [np.full((len(inputs["images"]), 512), value, dtype=np.float32)]

    session.run = run

    with pytest.raises(ValueError, match="invalid embeddings"):
        embedder.embed_crops([np.zeros((16, 8, 3), dtype=np.uint8)])
