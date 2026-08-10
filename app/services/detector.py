"""DEIMv2 object detection for soccer game-state reconstruction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
_INPUT_SIZE = 896


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


class DEIMv2Detector:
    """ONNX detector with graph-integrated DETR post-processing."""

    _IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    _IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: int = _INPUT_SIZE,
        confidence_threshold: float | None = None,
    ) -> None:
        import onnxruntime as ort

        from app.utils.cuda_env import get_providers
        from app.utils.models_registry import ensure_model

        path = (
            Path(model_path)
            if model_path is not None
            else ensure_model(
                "deimv2_detector",
                auto_download=settings.model_auto_download,
            )
        )
        if input_size < 1:
            raise ValueError("input_size must be positive")
        self.input_size = input_size
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.deimv2_confidence_threshold
        )

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path),
            options,
            providers=get_providers(prefer_gpu=True),
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise ValueError("DEIMv2 ONNX must expose image and target-size inputs")
        self._image_input = inputs[0].name
        self._size_input = inputs[1].name
        logger.info(
            "DEIMv2 loaded: path=%s input=%d threshold=%.3f providers=%s",
            path.name,
            self.input_size,
            self.confidence_threshold,
            self.session.get_providers(),
        )

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        size = self.input_size
        height, width = image.shape[:2]
        scale = min(size / width, size / height)
        resized_width = int(width * scale)
        resized_height = int(height * scale)
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (size - resized_width) // 2
        pad_y = (size - resized_height) // 2
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return rgb.transpose(2, 0, 1)[None], scale, pad_x, pad_y

    def detect(self, image: np.ndarray) -> list[Detection]:
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        target_size = np.asarray([[self.input_size, self.input_size]], dtype=np.int64)
        labels, boxes, scores = self.session.run(
            None,
            {
                self._image_input: tensor,
                self._size_input: target_size,
            },
        )

        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]
        height, width = image.shape[:2]
        keep = scores > self.confidence_threshold
        detections: list[Detection] = []
        for label, box, score in zip(labels[keep], boxes[keep], scores[keep], strict=True):
            class_id = int(label) - 1
            if class_id not in settings.class_names:
                continue

            x1 = float(np.clip((float(box[0]) - pad_x) / scale, 0.0, width))
            y1 = float(np.clip((float(box[1]) - pad_y) / scale, 0.0, height))
            x2 = float(np.clip((float(box[2]) - pad_x) / scale, 0.0, width))
            y2 = float(np.clip((float(box[3]) - pad_y) / scale, 0.0, height))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=settings.class_names[class_id],
                    confidence=float(score),
                )
            )
        return detections


_INSTANCE: DEIMv2Detector | None = None


def get_detector() -> DEIMv2Detector:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DEIMv2Detector()
    return _INSTANCE
