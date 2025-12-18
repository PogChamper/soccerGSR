# detector.py - заглушка для реализации

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Detection:
    """Результат детекции одного объекта."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

class YOLODetector:
    """TODO: Реализовать YOLO детектор."""

