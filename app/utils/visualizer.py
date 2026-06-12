import cv2
import numpy as np
from typing import List, Optional

from app.services.detector import Detection
from app.config import get_settings

settings = get_settings()


# Stable colors for the two teams (BGR). Referee/ball still use class colors.
TEAM_COLORS = {
    0: (255, 80, 0),     # team A: cyan-ish blue
    1: (0, 80, 255),     # team B: warm orange
}


def _color_for(det: Detection, extra: Optional[dict]) -> tuple:
    """Pick bbox color based on team for players, class color otherwise."""
    if extra and det.class_id == 0 and extra.get("team_id") in TEAM_COLORS:
        return TEAM_COLORS[extra["team_id"]]
    if extra and det.class_id == 1 and extra.get("team_id") in TEAM_COLORS:
        # Goalkeeper of team N — same hue, brighter
        c = TEAM_COLORS[extra["team_id"]]
        return tuple(min(255, int(v * 1.2) + 40) for v in c)
    return settings.class_colors.get(det.class_id, (255, 255, 255))


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    draw_labels: bool = True,
    draw_confidence: bool = True,
    line_thickness: int = 2,
    extras: Optional[List[dict]] = None,
) -> np.ndarray:
    """Draw bboxes + labels (track id, jersey, class, confidence) on a copy
    of the frame. ``extras[i]`` may carry track_id/jersey_number/team_id."""
    annotated = frame.copy()

    if extras is not None and len(extras) != len(detections):
        # length mismatch is fatal — caller bug; drop extras silently rather than raise
        extras = None

    for i, det in enumerate(detections):
        extra = extras[i] if extras else None
        color = _color_for(det, extra)

        x1, y1, x2, y2 = [int(c) for c in det.bbox]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        if draw_labels or draw_confidence or extra:
            label_parts = []
            if extra and extra.get("track_id") is not None:
                label_parts.append(f"#{extra['track_id']}")
            if extra and extra.get("jersey_number") is not None:
                label_parts.append(f"J{extra['jersey_number']}")
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
    """Draw the class-colour legend in the requested corner."""
    annotated = frame.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    max_text_width = 0
    for class_id, class_name in settings.class_names.items():
        (text_w, _), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_w)
    
    legend_width = max_text_width + 40  # color box + padding
    legend_height = len(settings.class_names) * item_height + padding * 2

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

