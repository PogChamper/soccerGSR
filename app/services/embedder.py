"""SoccerNet OSNet inference for tracking and offline association."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.utils.cuda_env import get_providers_with_options

logger = logging.getLogger(__name__)

_INPUT_HEIGHT = 256
_INPUT_WIDTH = 128
_EMBEDDING_DIM = 512
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)


class OSNetEmbedder:
    """Batched ONNX feature extractor with the SoccerNet OSNet contract."""

    def __init__(
        self,
        model_path: Path,
        *,
        prefer_gpu: bool = True,
        batch_size: int = 128,
    ) -> None:
        import onnxruntime as ort

        if not model_path.is_file():
            raise FileNotFoundError(f"OSNet model not found: {model_path}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        providers = get_providers_with_options(
            prefer_gpu=prefer_gpu,
            gpu_mem_limit_gb=2.0,
            cudnn_conv_algo_search="HEURISTIC",
        )
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("OSNet ONNX must expose one input and one output")

        output_dim = outputs[0].shape[-1]
        if isinstance(output_dim, int) and output_dim != _EMBEDDING_DIM:
            raise ValueError(f"OSNet output dimension must be {_EMBEDDING_DIM}, got {output_dim}")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._batch_size = batch_size
        logger.info(
            "OSNet loaded: path=%s providers=%s",
            model_path.name,
            self._session.get_providers(),
        )

    @property
    def embed_dim(self) -> int:
        return _EMBEDDING_DIM

    @staticmethod
    def _crop(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left = int(max(0.0, x1))
        top = int(max(0.0, y1))
        right = int(min(float(width), x2))
        bottom = int(min(float(height), y2))
        if right - left < 2 or bottom - top < 2:
            return np.zeros((8, 4, 3), dtype=np.uint8)
        return frame[top:bottom, left:right]

    @staticmethod
    def _preprocess(crops: Sequence[np.ndarray]) -> np.ndarray:
        batch = np.empty(
            (len(crops), 3, _INPUT_HEIGHT, _INPUT_WIDTH),
            dtype=np.float32,
        )
        for index, crop in enumerate(crops):
            image = cv2.resize(
                crop,
                (_INPUT_WIDTH, _INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            batch[index] = image.transpose(2, 0, 1)
        batch -= _MEAN
        batch /= _STD
        return batch

    def embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return L2-normalized features in input order."""
        if not crops:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(crops), self._batch_size):
            batch = self._preprocess(crops[start : start + self._batch_size])
            output = self._session.run(
                [self._output_name],
                {self._input_name: batch},
            )[0]
            features = np.asarray(output, dtype=np.float32)
            if features.ndim != 2 or features.shape != (len(batch), _EMBEDDING_DIM):
                raise ValueError(
                    f"OSNet output must have shape (batch, {_EMBEDDING_DIM}), got {features.shape}"
                )
            chunks.append(features)

        embeddings = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms <= 1e-12):
            raise ValueError("OSNet output contains invalid embeddings")
        np.divide(embeddings, norms, out=embeddings)
        return embeddings

    def embed_boxes(
        self,
        frame: np.ndarray,
        bboxes: Sequence[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Return one feature row per box in detection order."""
        if not bboxes:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
        return self.embed_crops([self._crop(frame, bbox) for bbox in bboxes])


_INSTANCE: OSNetEmbedder | None = None


def get_embedder(model_path: Path | None = None) -> OSNetEmbedder:
    global _INSTANCE
    if _INSTANCE is None:
        from app.config import get_settings
        from app.utils.models_registry import ensure_model

        path = model_path or ensure_model(
            "osnet_reid",
            auto_download=get_settings().model_auto_download,
        )
        _INSTANCE = OSNetEmbedder(path)
    return _INSTANCE
