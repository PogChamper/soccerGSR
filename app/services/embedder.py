"""DINOv3 ViT-S+/16 appearance embedder (ONNX, 384-d) for player crops.
Consumed by BoT-SORT (ReID association) and the track-merger. Process-wide
singleton with no per-clip state; pre-loaded from ``app.main.lifespan``
before anything imports torch (see the load-order note there).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.utils.cuda_env import get_providers_with_options

logger = logging.getLogger(__name__)


# ImageNet-style normalisation (matches DINOv3ViTImageProcessorFast defaults)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_INPUT_HW = (224, 224)


class DinoEmbedder:
    """Thin wrapper around an ONNX DINOv3 session (thread-safe)."""

    def __init__(
        self,
        model_path: Path,
        *,
        prefer_gpu: bool = True,
        batch_max: int = 64,
    ) -> None:
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(f"DINOv3 ONNX not found at {model_path}")

        providers = get_providers_with_options(
            prefer_gpu=prefer_gpu,
            gpu_mem_limit_gb=4.0,
            cudnn_conv_algo_search="HEURISTIC",
        )
        self._sess = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        out_shape = self._sess.get_outputs()[0].shape
        # static dim is whatever the second axis says (384 for ViT-S+/16)
        self._embed_dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 384
        self._batch_max = batch_max
        logger.info(
            f"embedder: DINOv3 loaded providers={self._sess.get_providers()} "
            f"path={model_path.name} dim={self._embed_dim}"
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _crop(frame_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1i = max(0, int(round(x1)))
        y1i = max(0, int(round(y1)))
        x2i = min(w, int(round(x2)))
        y2i = min(h, int(round(y2)))
        if x2i - x1i < 4 or y2i - y1i < 4:
            return None
        return frame_bgr[y1i:y2i, x1i:x2i]

    @staticmethod
    def _preprocess_batch(crops_bgr: List[np.ndarray]) -> np.ndarray:
        """BGR uint8 list -> NCHW float32, ImageNet normalised, 224x224."""
        n = len(crops_bgr)
        out = np.empty((n, 3, _INPUT_HW[0], _INPUT_HW[1]), dtype=np.float32)
        for i, c in enumerate(crops_bgr):
            r = cv2.resize(c, (_INPUT_HW[1], _INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
            r = r.astype(np.float32) / 255.0
            # HWC -> CHW
            out[i] = r.transpose(2, 0, 1)
        out -= _MEAN
        out /= _STD
        return out

    # ------------------------------------------------------------- API

    def embed_crops(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR crops -> (N, embed_dim) float32, l2-normalised.

        L2 normalisation is applied so cosine similarity == dot product;
        BoT-SORT's ``embedding_distance`` then matches its appearance_thresh
        scale.
        """
        if not crops_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        out_list: List[np.ndarray] = []
        for i in range(0, len(crops_bgr), self._batch_max):
            chunk = crops_bgr[i : i + self._batch_max]
            x = self._preprocess_batch(chunk)
            y = self._sess.run([self._output_name], {self._input_name: x})[0]
            out_list.append(y.astype(np.float32, copy=False))
        emb = np.concatenate(out_list, axis=0) if len(out_list) > 1 else out_list[0]

        # L2-normalise rows
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, np.maximum(norm, 1e-12), out=emb)
        return emb

    def embed_boxes(
        self,
        frame_bgr: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Per-bbox embedding aligned to input order. Empty/invalid boxes
        get a zero vector (so caller can still index by detection idx).
        """
        if not bboxes:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        crops: List[np.ndarray] = []
        good_idx: List[int] = []
        for i, b in enumerate(bboxes):
            c = self._crop(frame_bgr, b)
            if c is not None:
                crops.append(c)
                good_idx.append(i)

        out = np.zeros((len(bboxes), self._embed_dim), dtype=np.float32)
        if not crops:
            return out
        emb = self.embed_crops(crops)
        for k, i in enumerate(good_idx):
            out[i] = emb[k]
        return out


# --------------------------------------------------------- singleton

_INSTANCE: Optional[DinoEmbedder] = None


def get_embedder(model_path: Optional[Path] = None) -> DinoEmbedder:
    global _INSTANCE
    if _INSTANCE is None:
        from app.utils.models_registry import ensure_model

        path = model_path or ensure_model("dinov3_embedder")
        _INSTANCE = DinoEmbedder(path)
    return _INSTANCE


def reset_embedder() -> None:
    """Test helper — drops the cached instance."""
    global _INSTANCE
    _INSTANCE = None
