import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import cv2

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Detection:
    """Single detection result."""

    bbox: Tuple[float, float, float, float]     # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


class YOLODetector:
    """YOLOv5lu detector using ONNX Runtime."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        from app.utils.models_registry import ensure_model

        self.model_path = model_path or str(ensure_model("yolo_detector"))
        self.input_size = (
            input_size if input_size is not None else settings.model_input_size
        )
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None
            else settings.confidence_threshold
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else settings.iou_threshold
        )
        self.class_names = settings.class_names
        self.num_classes = len(self.class_names)

        self.session = self._load_model()

    def _load_model(self) -> ort.InferenceSession:
        """Initialize ONNX session, prefer CUDA on WSL/GPU."""
        from app.utils.cuda_env import get_providers

        providers = get_providers(prefer_gpu=True)
        session = ort.InferenceSession(self.model_path, providers=providers)
        return session

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """BGR (H, W, 3) -> letterboxed RGB tensor (1, 3, S, S) float32.

        Returns:
            - preprocessed tensor
            - scale ratio
            - (pad_x, pad_y) padding offsets
        """
        orig_h, orig_w = image.shape[:2]

        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        # e.g., if (orig_w, orig_h) = (1080, 1920) and input_size = 1280,
        # then scale = 1280/1920 = 0.667, (new_w, new_h) = (1280, 720), (pad_x, pad_y) = (0, 280)

        # Resize image to size (new_w, new_h)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (letterbox)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w, :] = resized

        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)    # BGR -> RGB

        tensor = padded.astype(np.float32) / 255.0      # [0; 255] -> [0.0; 1.0]
        tensor = tensor.transpose(2, 0, 1)              # (H, W, C) -> (C, H, W)
        tensor = tensor[None, ...]                      # (1, C, H, W)

        return tensor, scale, (pad_x, pad_y)
    
    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        padding: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> List[Detection]:
        """Raw output (1, 8, 33600) = 4 bbox + 4 class channels over three
        scales -> thresholded, un-letterboxed, NMS-filtered detections."""
        predictions = output[0, ...].T          # (33600, 8)
        boxes = predictions[:, :4]              # (33600, 4)
        class_scores = predictions[:, 4:]       # (33600, 4)

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_ids.size), class_ids]    # (33600,)

        mask = confidences >= self.conf_threshold
        if (~mask).all():
            return []

        boxes = boxes[mask, :]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert xywh format to xyxy
        xc, yc, w, h = boxes.T
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0

        # Rescale xyxy to original size
        pad_x, pad_y = padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Clip coords
        orig_h, orig_w = original_size
        x1 = np.clip(x1, 0.0, orig_w)
        y1 = np.clip(y1, 0.0, orig_h)
        x2 = np.clip(x2, 0.0, orig_w)
        y2 = np.clip(y2, 0.0, orig_h)

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        # Build detection results
        detections = []
        for idx in indices:
            detection = Detection(
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                class_id=int(class_ids[idx]),
                class_name=self.class_names.get(int(class_ids[idx]), "unknown"),
                confidence=float(confidences[idx])
            )
            detections.append(detection)

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """Greedy NMS over xyxy boxes; returns indices to keep."""
        if boxes.shape[0] == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        area = (x2 - x1) * (y2 - y1)
        order = np.argsort(confidences)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break
            
            rest = order[1:]

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            if intersection.sum() == 0:
                order = rest
                continue

            union = area[i] + area[rest] - intersection
            iou = intersection / union

            order = rest[iou <= iou_threshold]

        return keep

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects on a single BGR image."""
        tensor, scale, padding = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: tensor})[0]
        return self.postprocess(output, scale, padding, image.shape[:2])


class DEIMv2Detector:
    """DEIMv2-DINOv3 detector (DETR-style) using ONNX Runtime.

    The ONNX graph bakes in DETR post-processing, so there is no anchor decode
    and no NMS: it emits a fixed top-300 candidate set as ``(labels, boxes,
    scores)`` and we keep candidates above a single score threshold.

    Differences vs the YOLO path:
    * preprocessing = aspect-preserving resize + zero-pad to a square +
      ImageNet normalisation (for s/m/l/x sizes);
    * the second graph input ``orig_target_sizes`` is fed the padded input
      size ``[[size, size]]`` so boxes come back in padded-input coordinates,
      which we then un-pad / un-scale back to original-frame pixels;
    * the model was trained with ``num_classes=5`` where index 0 is background,
      so the service ``cls_id = label - 1`` (0=player .. 3=ball).
    """

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _NO_NORM_SIZES = {"atto", "femto", "pico", "n"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
        model_size: Optional[str] = None,
    ):
        from app.utils.cuda_env import get_providers
        from app.utils.models_registry import ensure_model

        self.model_path = model_path or str(ensure_model("deimv2_detector"))
        self.input_size = (
            input_size if input_size is not None else settings.deimv2_input_size
        )
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None
            else settings.deimv2_confidence_threshold
        )
        self.model_size = (model_size or settings.deimv2_model_size).lower()
        self.class_names = settings.class_names
        self._apply_norm = self.model_size not in self._NO_NORM_SIZES

        providers = get_providers(prefer_gpu=True)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            self.model_path, sess_opts, providers=providers
        )
        names = [i.name for i in self.session.get_inputs()]
        # first input is the image tensor, second is orig_target_sizes
        self._img_in = names[0]
        self._sizes_in = names[1] if len(names) > 1 else "orig_target_sizes"
        logger.info(
            f"detector: DEIMv2 {self.model_size}@{self.input_size} "
            f"thr={self.conf_threshold} providers={self.session.get_providers()} "
            f"path={self.model_path}"
        )

    def _preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        """BGR uint8 (H,W,3) -> (1,3,S,S) float32, plus (ratio, pad_x, pad_y)."""
        size = self.input_size
        h, w = image.shape[:2]
        ratio = min(size / w, size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._apply_norm:
            rgb = (rgb - self._IMAGENET_MEAN) / self._IMAGENET_STD
        tensor = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)
        return tensor, ratio, pad_x, pad_y

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR image -> List[Detection]."""
        size = self.input_size
        tensor, ratio, pad_x, pad_y = self._preprocess(image)
        orig_target_sizes = np.array([[size, size]], dtype=np.int64)

        labels, boxes, scores = self.session.run(
            None,
            {self._img_in: tensor, self._sizes_in: orig_target_sizes},
        )
        labels, boxes, scores = labels[0], boxes[0], scores[0]

        orig_h, orig_w = image.shape[:2]
        detections: List[Detection] = []
        keep = scores > self.conf_threshold
        for label, box, score in zip(labels[keep], boxes[keep], scores[keep]):
            cls_id = int(label) - 1               # drop background@0 -> service ids
            if cls_id < 0 or cls_id not in self.class_names:
                continue
            x1 = (float(box[0]) - pad_x) / ratio
            y1 = (float(box[1]) - pad_y) / ratio
            x2 = (float(box[2]) - pad_x) / ratio
            y2 = (float(box[3]) - pad_y) / ratio
            x1 = float(np.clip(x1, 0.0, orig_w))
            y1 = float(np.clip(y1, 0.0, orig_h))
            x2 = float(np.clip(x2, 0.0, orig_w))
            y2 = float(np.clip(y2, 0.0, orig_h))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, "unknown"),
                    confidence=float(score),
                )
            )
        return detections


_detector_instance = None


def get_detector():
    """Get or create the configured detector instance.

    Backend chosen by ``settings.detector_backend`` ("deimv2" | "yolo").
    Both backends expose the same ``.detect(frame) -> List[Detection]`` API.
    """
    global _detector_instance
    if _detector_instance is None:
        backend = (settings.detector_backend or "deimv2").lower()
        if backend == "yolo":
            _detector_instance = YOLODetector()
        else:
            _detector_instance = DEIMv2Detector()
    return _detector_instance


def reset_detector() -> None:
    """Test helper — drops the cached instance (e.g. after switching backend)."""
    global _detector_instance
    _detector_instance = None
