"""Top-down minimap renderer, overlaid on the output frame.

World coords are centred (x in [-52.5, 52.5], y in [-34, 34]). Markers:
circle = player, triangle = GK, diamond = referee, white dot = ball; jersey
numbers get a contrast halo. Drawn on a supersampled canvas and downscaled
once with INTER_AREA — that is what keeps thin lines/text crisp after video
re-encoding.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from app.services.clip_state import FrameInfo, FrameObservation, TrackInfo
from app.utils.visualizer import TEAM_COLORS

logger = logging.getLogger(__name__)


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
DEFAULT_W = 480
DEFAULT_H = 311             # 480 * 68/105 ≈ 310.9
MARGIN_M = 2.0

# Internal supersampling: draw big, downscale once -> crisp edges/text.
SUPERSAMPLE = 3

# How many consecutive uncalibrated frames to keep showing the last good map
# before falling back to a "no calibration" placeholder (~2 s at 25 fps).
FREEZE_MAX_FRAMES = 50

# Marker sizes are in FINAL (post-downscale) pixels; scaled by SUPERSAMPLE
# internally while drawing.
PLAYER_RADIUS = 8           # filled circle
GK_HALF_SIZE = 9            # half-width of triangle
REF_HALF_SIZE = 8           # half-diagonal of diamond
BALL_RADIUS = 4
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_THICK = 1.5         # final px

# Jersey number text (final px / scale)
NUM_SCALE = 0.36
NUM_FG_THICK = 1.0
NUM_HALO_THICK = 2.4

LINE_COLOR = (236, 236, 236)
LINE_THICK = 1.4            # final px
# Two-tone mowing stripes (BGR)
STRIPE_A = (48, 124, 58)
STRIPE_B = (42, 112, 52)
N_STRIPES = 12

GK_COLOR_FALLBACK = (60, 230, 240)   # light cyan
REF_COLOR = (60, 200, 250)           # bright yellow-orange (BGR)


def _bgr_luminance(bgr: Tuple[int, int, int]) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def _text_color_for(fill_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Pick black or white text so it contrasts against ``fill_bgr``."""
    return (0, 0, 0) if _bgr_luminance(fill_bgr) > 140 else (255, 255, 255)


def _draw_text_with_halo(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    *,
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    scale: float = 0.5,
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    halo_color: Tuple[int, int, int] = (0, 0, 0),
    halo_thickness: int = 3,
    fg_thickness: int = 1,
) -> None:
    """Render text with a thicker halo for legibility on any background."""
    cv2.putText(img, text, org, font, scale, halo_color, halo_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, fg_color, fg_thickness, cv2.LINE_AA)


def _draw_pitch(rw: int, rh: int, ss: int) -> np.ndarray:
    """Broadcast-style pitch at render resolution (rw, rh).

    ``ss`` is the supersample factor used to scale physical line thickness.
    """
    img = np.empty((rh, rw, 3), dtype=np.uint8)
    # mowing stripes (vertical bands)
    stripe_w = rw / N_STRIPES
    for i in range(N_STRIPES):
        x0 = int(round(i * stripe_w))
        x1 = int(round((i + 1) * stripe_w))
        img[:, x0:x1] = STRIPE_A if (i % 2 == 0) else STRIPE_B

    def m2px(x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(round((x_m + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * rw))
        py = int(round((y_m + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * rh))
        return px, py

    th = max(1, int(round(LINE_THICK * ss)))
    aa = cv2.LINE_AA

    def m2r(d_m: float) -> int:
        return int(round(d_m / PITCH_LENGTH_M * rw))

    half_l, half_w = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
    # outer boundary
    cv2.rectangle(img, m2px(-half_l, -half_w), m2px(half_l, half_w), LINE_COLOR, th, aa)
    # halfway line
    cv2.line(img, m2px(0, -half_w), m2px(0, half_w), LINE_COLOR, th, aa)
    # centre circle + spot
    cv2.circle(img, m2px(0, 0), m2r(9.15), LINE_COLOR, th, aa)
    cv2.circle(img, m2px(0, 0), max(2, int(round(1.3 * ss))), LINE_COLOR, -1, aa)
    # penalty boxes (16.5m deep, 40.32m wide)
    cv2.rectangle(img, m2px(-half_l, -20.16), m2px(-half_l + 16.5, 20.16), LINE_COLOR, th, aa)
    cv2.rectangle(img, m2px(half_l - 16.5, -20.16), m2px(half_l, 20.16), LINE_COLOR, th, aa)
    # 6-yard boxes (5.5m deep, 18.32m wide)
    cv2.rectangle(img, m2px(-half_l, -9.16), m2px(-half_l + 5.5, 9.16), LINE_COLOR, th, aa)
    cv2.rectangle(img, m2px(half_l - 5.5, -9.16), m2px(half_l, 9.16), LINE_COLOR, th, aa)
    # penalty spots (11m from goal line)
    spot_r = max(2, int(round(1.2 * ss)))
    cv2.circle(img, m2px(-half_l + 11.0, 0), spot_r, LINE_COLOR, -1, aa)
    cv2.circle(img, m2px(half_l - 11.0, 0), spot_r, LINE_COLOR, -1, aa)
    # penalty arcs (the "D"): radius 9.15 around the spot, only the part
    # outside the box. cos(theta) = (16.5-11)/9.15 ≈ 0.601 -> ~53.1 deg.
    arc_r = m2r(9.15)
    cv2.ellipse(img, m2px(-half_l + 11.0, 0), (arc_r, arc_r), 0, -53, 53, LINE_COLOR, th, aa)
    cv2.ellipse(img, m2px(half_l - 11.0, 0), (arc_r, arc_r), 0, 127, 233, LINE_COLOR, th, aa)
    return img


# ---- shape primitives (drawn at render resolution) ------------------------


def _draw_circle(img, center, radius, fill, outline=OUTLINE_COLOR, thick=2):
    cv2.circle(img, center, radius, fill, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, radius, outline, thick, lineType=cv2.LINE_AA)


def _draw_triangle(img, center, half, fill, outline=OUTLINE_COLOR, thick=2):
    cx, cy = center
    pts = np.array([
        [cx, cy - half],
        [cx - half, cy + int(half * 0.85)],
        [cx + half, cy + int(half * 0.85)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], fill, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], True, outline, thick, lineType=cv2.LINE_AA)


def _draw_diamond(img, center, half, fill, outline=OUTLINE_COLOR, thick=2):
    cx, cy = center
    pts = np.array([
        [cx, cy - half],
        [cx + half, cy],
        [cx, cy + half],
        [cx - half, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], fill, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts], True, outline, thick, lineType=cv2.LINE_AA)


# ---- renderer -------------------------------------------------------------


class MinimapRenderer:
    def __init__(
        self,
        *,
        width: int = DEFAULT_W,
        height: int = DEFAULT_H,
        position: str = "bottom-right",
        margin_px: int = 14,
        supersample: int = SUPERSAMPLE,
    ):
        self.width = width
        self.height = height
        self.position = position
        self.margin_px = margin_px
        self.ss = max(1, int(supersample))
        self.rw = width * self.ss
        self.rh = height * self.ss
        self._pitch_template = _draw_pitch(self.rw, self.rh, self.ss)
        # last drawn pixel per track_id (render-resolution) — used for 1-px snap
        self._last_px: Dict[int, Tuple[int, int]] = {}
        # last successfully rendered (final-res) minimap, frozen during gaps
        self._last_mini: np.ndarray | None = None
        self._frozen_count: int = 0
        # precomputed render-res marker sizes/thicknesses
        self._r_player = max(2, int(round(PLAYER_RADIUS * self.ss)))
        self._r_gk = max(2, int(round(GK_HALF_SIZE * self.ss)))
        self._r_ref = max(2, int(round(REF_HALF_SIZE * self.ss)))
        self._r_ball = max(2, int(round(BALL_RADIUS * self.ss)))
        self._outline = max(1, int(round(OUTLINE_THICK * self.ss)))
        self._num_scale = NUM_SCALE * self.ss
        self._num_fg = max(1, int(round(NUM_FG_THICK * self.ss)))
        self._num_halo = max(self._num_fg + 1, int(round(NUM_HALO_THICK * self.ss)))

    def reset_clip_state(self) -> None:
        """Drop per-clip state (snap cache, frozen frame).

        The renderer instance is shared across jobs (process-wide singleton),
        but ``_last_px`` / ``_last_mini`` / ``_frozen_count`` are clip-scoped:
        track ids restart from 1 for every clip, so stale entries from a
        previous job would cause wrong position snapping and a leaked
        \"frozen\" minimap on the first uncalibrated frames. Call this at the
        start of every render pass.
        """
        self._last_px = {}
        self._last_mini = None
        self._frozen_count = 0

    def _m2px(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """World metres -> render-resolution pixel."""
        px = int(round((x_m + PITCH_LENGTH_M / 2) / PITCH_LENGTH_M * self.rw))
        py = int(round((y_m + PITCH_WIDTH_M / 2) / PITCH_WIDTH_M * self.rh))
        return px, py

    def render(
        self,
        observations: Iterable[FrameObservation],
        tracks: Dict[int, TrackInfo],
    ) -> np.ndarray:
        img = self._pitch_template.copy()

        # Draw in z-order: players first, then GK, then ref, ball on top.
        def _disp_cls(o: FrameObservation) -> int:
            if o.display_cls is not None:
                return o.display_cls
            tr = tracks.get(o.track_id) if o.track_id is not None else None
            return tr.cls_id if tr is not None else o.cls_id

        def _z(o: FrameObservation) -> int:
            return {0: 0, 1: 1, 2: 2, 3: 4}.get(_disp_cls(o), 0)

        ordered = sorted(
            (o for o in observations if o.pitch_xy is not None), key=_z
        )

        last_px = self._last_px
        snap_tol = self.ss  # ~1 final px

        for obs in ordered:
            x_m, y_m = obs.pitch_xy
            px, py = self._m2px(x_m, y_m)
            if not (-40 <= px <= self.rw + 40 and -40 <= py <= self.rh + 40):
                continue

            if obs.track_id is not None:
                prev = last_px.get(obs.track_id)
                if prev is not None and abs(px - prev[0]) <= snap_tol and abs(py - prev[1]) <= snap_tol:
                    px, py = prev
                last_px[obs.track_id] = (px, py)

            track = tracks.get(obs.track_id) if obs.track_id is not None else None
            # space-time-voted class -> stable shape even across tracker ID swaps
            cls_id = _disp_cls(obs)
            jersey = track.jersey_number if track is not None else None

            if cls_id == 3:                                    # ball
                _draw_circle(img, (px, py), self._r_ball, (255, 255, 255), thick=self._outline)
                continue

            if cls_id == 2:                                    # referee
                _draw_diamond(img, (px, py), self._r_ref, REF_COLOR, thick=self._outline)
                continue

            fill = (180, 180, 180)
            if cls_id == 1:                                    # goalkeeper
                fill = TEAM_COLORS.get(obs.team_id, GK_COLOR_FALLBACK)
                _draw_triangle(img, (px, py), self._r_gk, fill, thick=self._outline)
            else:                                              # outfield (cls 0)
                if track is not None and track.team_label == "referee":
                    _draw_diamond(img, (px, py), self._r_ref, REF_COLOR, thick=self._outline)
                    continue
                fill = TEAM_COLORS.get(obs.team_id, (180, 180, 180))
                _draw_circle(img, (px, py), self._r_player, fill, thick=self._outline)

            # jersey number (players + GK)
            if jersey is not None:
                txt = str(int(jersey))
                fg = _text_color_for(fill)
                halo = (0, 0, 0) if fg == (255, 255, 255) else (255, 255, 255)
                size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, self._num_scale, self._num_fg)[0]
                tx = px - size[0] // 2
                ty = py + size[1] // 2
                _draw_text_with_halo(
                    img, txt, (tx, ty),
                    scale=self._num_scale, fg_color=fg, halo_color=halo,
                    halo_thickness=self._num_halo, fg_thickness=self._num_fg,
                )

        # single high-quality downscale -> crisp edges in the encoded video
        if self.ss != 1:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img

    def overlay(
        self,
        frame: np.ndarray,
        frame_info: FrameInfo,
        observations: List[FrameObservation],
        tracks: Dict[int, TrackInfo],
    ) -> np.ndarray:
        if not observations or all(o.pitch_xy is None for o in observations):
            # Calibration unavailable this frame (e.g. close-up). Rather than
            # blanking the map, freeze the last good render for a short window
            # so markers don't disappear; only after that show a placeholder.
            if self._last_mini is not None and self._frozen_count < FREEZE_MAX_FRAMES:
                mini = self._last_mini.copy()
                self._frozen_count += 1
                cv2.putText(
                    mini, "tracking...", (8, self.height - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA,
                )
            else:
                mini = cv2.resize(
                    self._pitch_template, (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                ) if self.ss != 1 else self._pitch_template.copy()
                cv2.putText(
                    mini, "no calibration", (8, self.height - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )
        else:
            mini = self.render(observations, tracks)
            self._last_mini = mini.copy()
            self._frozen_count = 0

        h, w = frame.shape[:2]
        if self.position == "bottom-right":
            x0 = w - self.width - self.margin_px
            y0 = h - self.height - self.margin_px
        elif self.position == "bottom-left":
            x0 = self.margin_px
            y0 = h - self.height - self.margin_px
        elif self.position == "top-right":
            x0 = w - self.width - self.margin_px
            y0 = self.margin_px
        else:
            x0 = self.margin_px
            y0 = self.margin_px

        # semi-transparent dark plate behind for readability
        pad = 6
        plate_x0 = max(0, x0 - pad)
        plate_y0 = max(0, y0 - pad)
        plate_x1 = min(w, x0 + self.width + pad)
        plate_y1 = min(h, y0 + self.height + pad)
        plate = frame[plate_y0:plate_y1, plate_x0:plate_x1]
        if plate.size > 0:
            plate[...] = (plate * 0.35).astype(np.uint8)

        frame[y0:y0 + self.height, x0:x0 + self.width] = mini

        # subtle white outline around the minimap
        cv2.rectangle(
            frame,
            (x0 - 1, y0 - 1),
            (x0 + self.width, y0 + self.height),
            (220, 220, 220), 1, cv2.LINE_AA,
        )
        return frame


_INSTANCE = None


def get_minimap_renderer() -> MinimapRenderer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MinimapRenderer()
    return _INSTANCE


def reset_minimap_renderer() -> None:
    global _INSTANCE
    _INSTANCE = None
