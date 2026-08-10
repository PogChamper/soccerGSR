import cv2
import numpy as np

from app.config import get_settings
from app.services.detector import Detection

settings = get_settings()


# Stable colors for the two teams (BGR). Referee/ball still use class colors.
TEAM_COLORS = {
    0: (255, 80, 0),  # team A: cyan-ish blue
    1: (0, 80, 255),  # team B: warm orange
}


def _color_for(det: Detection, extra: dict | None) -> tuple[int, int, int]:
    """Pick bbox color based on team for players, class color otherwise."""
    if extra and det.class_id == 0 and extra.get("team_id") in TEAM_COLORS:
        return TEAM_COLORS[extra["team_id"]]
    if extra and det.class_id == 1 and extra.get("team_id") in TEAM_COLORS:
        # Goalkeeper of team N - same hue, brighter
        c = TEAM_COLORS[extra["team_id"]]
        return tuple(min(255, int(v * 1.2) + 40) for v in c)
    return settings.class_colors.get(det.class_id, (255, 255, 255))


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    extras: list[dict] | None = None,
) -> np.ndarray:
    """Draw detections and identity attributes in place."""
    if extras is not None and len(extras) != len(detections):
        raise ValueError("extras and detections must have the same length")

    for i, det in enumerate(detections):
        extra = extras[i] if extras else None
        color = _color_for(det, extra)

        x1, y1, x2, y2 = [int(c) for c in det.bbox]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        if extra and extra.get("track_id") is not None:
            label_parts.append(f"#{extra['track_id']}")
        if extra and extra.get("jersey_number") is not None:
            label_parts.append(f"J{extra['jersey_number']}")
        label_parts.append(det.class_name)
        label_parts.append(f"{det.confidence:.2f}")
        label = " ".join(label_parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        label_y1 = max(y1 - text_h - 10, 0)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)

        text_color = (0, 0, 0) if sum(color) > 384 else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - 5),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw the class-colour legend in the top-left corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    padding = 10
    item_height = 25

    max_text_width = 0
    for class_id, class_name in settings.class_names.items():
        (text_w, _), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
        max_text_width = max(max_text_width, text_w)

    legend_width = max_text_width + 40  # color box + padding
    legend_height = len(settings.class_names) * item_height + padding * 2

    x, y = padding, padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, (class_id, class_name) in enumerate(settings.class_names.items()):
        color = settings.class_colors.get(class_id, (255, 255, 255))
        item_y = y + padding + i * item_height

        cv2.rectangle(frame, (x + 5, item_y + 2), (x + 20, item_y + item_height - 5), color, -1)
        cv2.putText(
            frame,
            class_name,
            (x + 28, item_y + item_height - 8),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame
