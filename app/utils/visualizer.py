import cv2
import numpy as np
from typing import List

from app.services.detector import Detection
from app.config import get_settings

settings = get_settings()


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    draw_labels: bool = True,
    draw_confidence: bool = True,
    line_thickness: int = 2
) -> np.ndarray:
    """Draw detection bounding boxes on a frame.
    
    Args:
        frame: Input frame in BGR format
        detections: List of Detection objects
        draw_labels: Whether to draw class labels
        draw_confidence: Whether to draw confidence scores
        line_thickness: Thickness of bbox lines
        
    Returns:
        Frame with drawn detections
    """
    annotated = frame.copy()
    
    for det in detections:
        # Get color for this class
        color = settings.class_colors.get(det.class_id, (255, 255, 255))
        
        # Extract bbox coordinates
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
        
        if draw_labels or draw_confidence:
            # Prepare label text
            label_parts = []
            if draw_labels:
                label_parts.append(det.class_name)
            if draw_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            label = " ".join(label_parts)
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw label background
            label_y1 = max(y1 - text_h - 10, 0)
            label_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 6, label_y2),
                color,
                -1  # Filled
            )
            
            # Draw label text
            text_color = (0, 0, 0) if sum(color) > 384 else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 5),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
            
    return annotated


def draw_legend(
    frame: np.ndarray,
    position: str = "top-left",
    padding: int = 10,
    item_height: int = 25
) -> np.ndarray:
    """Draw class legend on frame.
    
    Args:
        frame: Input frame
        position: Legend position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        padding: Padding from edges
        item_height: Height of each legend item
        
    Returns:
        Frame with legend
    """
    annotated = frame.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Calculate legend dimensions
    max_text_width = 0
    for class_id, class_name in settings.class_names.items():
        (text_w, _), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_w)
    
    legend_width = max_text_width + 40  # color box + padding
    legend_height = len(settings.class_names) * item_height + padding * 2
    
    # Calculate position
    h, w = frame.shape[:2]
    if position == "top-left":
        x, y = padding, padding
    elif position == "top-right":
        x, y = w - legend_width - padding, padding
    elif position == "bottom-left":
        x, y = padding, h - legend_height - padding
    else:  # bottom-right
        x, y = w - legend_width - padding, h - legend_height - padding
    
    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Draw legend items
    for i, (class_id, class_name) in enumerate(settings.class_names.items()):
        color = settings.class_colors.get(class_id, (255, 255, 255))
        item_y = y + padding + i * item_height
        
        # Color box
        cv2.rectangle(
            annotated,
            (x + 5, item_y + 2),
            (x + 20, item_y + item_height - 5),
            color,
            -1
        )
        
        # Class name
        cv2.putText(
            annotated,
            class_name,
            (x + 28, item_y + item_height - 8),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
        
    return annotated


def draw_stats(
    frame: np.ndarray,
    stats: dict,
    position: str = "top-right"
) -> np.ndarray:
    """Draw detection stats on frame.
    
    Args:
        frame: Input frame
        stats: Dictionary with stats (e.g., {"fps": 30, "detections": 15})
        position: Stats position
        
    Returns:
        Frame with stats overlay
    """
    annotated = frame.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    lines = [f"{k}: {v}" for k, v in stats.items()]
    
    # Calculate dimensions
    max_width = 0
    line_height = 20
    for line in lines:
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_width = max(max_width, text_w)
    
    box_width = max_width + 20
    box_height = len(lines) * line_height + 15
    
    h, w = frame.shape[:2]
    if "right" in position:
        x = w - box_width - 10
    else:
        x = 10
    if "bottom" in position:
        y = h - box_height - 10
    else:
        y = 10
    
    # Draw background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Draw text
    for i, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (x + 10, y + 18 + i * line_height),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
        
    return annotated

