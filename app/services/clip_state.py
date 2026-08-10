"""In-memory state shared by the two processing passes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(slots=True)
class FrameObservation:
    """One detection of one track in one frame."""

    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    cls_id: int  # player=0, goalkeeper=1, referee=2, ball=3
    det_confidence: float
    track_id: int | None = None
    visibility_p: float | None = None
    ocr_logits_tens: list[float] | None = None
    ocr_logits_units: list[float] | None = None
    foot_xy_image: tuple[float, float] | None = None
    pitch_xy: tuple[float, float] | None = None
    team_id: int | None = None
    synthetic: bool = False


@dataclass(slots=True)
class FrameInfo:
    """Calibration evidence and result for one frame."""

    frame_idx: int
    width: int
    height: int
    # PnLCalib label -> normalized coords: x/y/p for keypoints, x_N/y_N/p_N for lines.
    keypoints: dict[int, dict[str, float]] = field(default_factory=dict)
    lines: dict[int, dict[str, float]] = field(default_factory=dict)
    homography_world_to_image: list[list[float]] | None = None
    homography_image_to_world: list[list[float]] | None = None
    homography_source: Literal["solved", "held", "interp", "none"] = "none"


@dataclass(slots=True)
class TrackInfo:
    """Per-track aggregated info, filled in the Aggregate stage."""

    track_id: int
    cls_id: int  # majority class
    cls_name: str
    team_id: int | None = None
    team_label: str | None = None
    jersey_number: int | None = None
    jersey_confidence: float | None = None
    n_frames_visible_gate: int = 0
    first_frame: int = -1
    last_frame: int = -1
    embedding_mean: np.ndarray | None = None
    n_observations: int = 0
    jersey_logits_tens: list[float] | None = None
    jersey_logits_units: list[float] | None = None
    jersey_vote_count: int = 0


@dataclass(slots=True)
class ClipMeta:
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    size_mb: float
    codec: str = ""


@dataclass(slots=True)
class ClipState:
    meta: ClipMeta
    frames: list[FrameInfo] = field(default_factory=list)
    observations: list[FrameObservation] = field(default_factory=list)
    tracks: dict[int, TrackInfo] = field(default_factory=dict)

    def observations_by_track(self) -> dict[int, list[FrameObservation]]:
        out: dict[int, list[FrameObservation]] = {}
        for obs in self.observations:
            if obs.track_id is None:
                continue
            out.setdefault(obs.track_id, []).append(obs)
        return out

    def observations_by_frame(self) -> dict[int, list[FrameObservation]]:
        out: dict[int, list[FrameObservation]] = {}
        for obs in self.observations:
            out.setdefault(obs.frame_idx, []).append(obs)
        return out

    def to_gsr_json(self) -> dict[str, Any]:
        """Serialize for /jobs/{id}/gsr.json."""
        meta = asdict(self.meta)
        frames = []
        for f in self.frames:
            frames.append(
                {
                    "frame_idx": f.frame_idx,
                    "width": f.width,
                    "height": f.height,
                    "n_keypoints": len(f.keypoints),
                    "n_lines": len(f.lines),
                    "has_calibration": f.homography_world_to_image is not None,
                    "h_source": f.homography_source,
                    "H_world2img": f.homography_world_to_image,
                    "H_img2world": f.homography_image_to_world,
                }
            )

        observations = []
        for obs in self.observations:
            track = self.tracks.get(obs.track_id)
            observations.append(
                {
                    "frame_idx": obs.frame_idx,
                    "track_id": obs.track_id,
                    "cls_id": track.cls_id if track is not None else obs.cls_id,
                    "raw_cls_id": obs.cls_id,
                    "det_confidence": obs.det_confidence,
                    "bbox_xyxy": list(obs.bbox_xyxy),
                    "visibility_p": obs.visibility_p,
                    "team_id": obs.team_id,
                    "pitch_xy": list(obs.pitch_xy) if obs.pitch_xy else None,
                    "synthetic": obs.synthetic,
                }
            )

        tracks = {
            str(tid): {
                "track_id": t.track_id,
                "cls_id": t.cls_id,
                "cls_name": t.cls_name,
                "team_id": t.team_id,
                "team_label": t.team_label,
                "jersey_number": t.jersey_number,
                "jersey_confidence": t.jersey_confidence,
                "n_frames_visible_gate": t.n_frames_visible_gate,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
            }
            for tid, t in self.tracks.items()
        }

        return {
            "meta": meta,
            "frames": frames,
            "observations": observations,
            "tracks": tracks,
        }
