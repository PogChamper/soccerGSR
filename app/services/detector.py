from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import cv2

from app.config import get_settings

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
        model_path: str = None,
        input_size: int = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        
    ):
        """Initialize detector with ONNX model.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Size of model input square image
            conf_threshold: Confidence threshold for postprocessing
            iou_threshold: IoU threshold for postprocessing
        """
        self.model_path = model_path or settings.model_path
        self.input_size = input_size or settings.model_input_size
        self.conf_threshold = conf_threshold or settings.confidence_threshold
        self.iou_threshold = iou_threshold or settings.iou_threshold
        self.class_names = settings.class_names
        self.num_classes = len(self.class_names)

        self.session = self._load_model()

    def _load_model(self) -> ort.InferenceSession:
        """Initialize ONNX session."""
        providers = []

        # Add GPU first if available
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')

        # Add CPU (with lower priority if GPU available)
        providers.append('CPUExecutionProvider')

        session = ort.InferenceSession(self.model_path, providers=providers)
        return session

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess input image for YOLO inference.
        
        Args:
            image: Input image of shape (H, W, 3) in BGR format, dtype=uint8

        Returns:
            - preprocessed tensor of shape (1, 3, input_size, input_size) in RGB format, dtype=float32
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
        """Postprocess YOLO output to get detections.
        
        Args:
            output: Raw model output of shape (1, 8, 33600), where
                - 8 = 4 (bbox: xc, yc, w, h) + 4 (classes)
                - 33600 = 1600 + 6400 + 25600 (concatenation of predictions across all three scales)
            scale: Scale ratio used in preprocessing
            padding: (pad_x, pad_y) letterbox padding offsets
            original_size: Original image (height, width)

        Returns:
            List of Detection objects
        """
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
        """Perform Non-Maximum Suppression (NMS) on prediction results.
        
        Args:
            boxes: Boxes array of shape (N, 4) in xyxy format
            confidences: Confidence scores array of shape (N,)
            iou_threshold: IoU threshold

        Returns:
            List of box indices to keep
        """
        if boxes.shape[0] == 0:
            return []
        
        x1, y1, x2, y2 = boxes.T
        
        # Calculate all boxes areas
        area = (x2 - x1) * (y2 - y1)
        
        # Get box indices sorted by confidences
        order = np.argsort(confidences)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break
            
            rest = order[1:]

            # Calculate IoU for current box and the remaining boxes
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

            # Keep boxes with IoU <= threshold for next iteration
            order = rest[iou <= iou_threshold]

        return keep
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a single imagge.
        
        Args:
            image: Input image of shape (H, W, 3) in BGR format
            
        Returns:
            List of Detection objects
        """
        # Preprocess
        tensor, scale, padding = self.preprocess(image)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: tensor})[0]

        # Postprocess
        detections = self.postprocess(
            output, 
            scale, 
            padding, 
            image.shape[:2]
        )

        return detections
    

# Singleton detector instance
_detector_instance: Optional[YOLODetector] = None


def get_detector() -> YOLODetector:
    """Get or create detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector()
    return _detector_instance
