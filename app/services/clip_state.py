"""Central data model for the GSR pipeline.

Pass1 fills in per-frame raw observations.
Aggregate fills in per-track and per-frame derived fields (team, jersey, calibration).
Pass2 reads everything and renders annotated video + minimap + gsr.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------- Per-frame, per-detection -------------

@dataclass
class FrameObservation:
    """One detection of one track in one frame."""

    frame_idx: int
    bbox_xyxy: Tuple[float, float, float, float]
    cls_id: int                                          # 0 player, 1 gk, 2 ref, 3 ball
    det_confidence: float
    track_id: Optional[int] = None
    visibility_p: Optional[float] = None                 # P(jersey number visible)
    ocr_logits_tens: Optional[List[float]] = None        # length 10 if computed
    ocr_logits_units: Optional[List[float]] = None
    foot_xy_image: Optional[Tuple[float, float]] = None  # midpoint of bbox bottom edge
    pitch_xy: Optional[Tuple[float, float]] = None       # meters, after calibration
    team_id: Optional[int] = None                        # 0/1 for players, set in aggregate
    synthetic: bool = False                              # interpolated gap-fill (minimap only)
    display_cls: Optional[int] = None                   # space-time-voted class for rendering


@dataclass
class FrameInfo:
    """All info about one video frame (besides per-track observations)."""

    frame_idx: int
    width: int
    height: int
    keypoints: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    """{kp_label: (x_norm, y_norm, p)} for kp_label in 1..73 (PnLCalib indexing).
    Coordinates are normalized to [0,1] relative to the HRNet input; the
    calibrator un-scales them via FramebyFrameCalib(denormalize=True)."""
    lines: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    """{line_label: (x_norm, y_norm, p)} extremities, normalized like keypoints."""
    cam_params: Optional[Dict[str, Any]] = None
    """PnLCalib-style dict: x_focal_length, y_focal_length, principal_point,
    position_meters, rotation_matrix."""
    homography_world_to_image: Optional[List[List[float]]] = None
    """3x3 H mapping world (x,y,1) on z=0 plane -> image (u,v,1)."""
    homography_image_to_world: Optional[List[List[float]]] = None


# ------------- Per-track aggregates -------------

@dataclass
class TrackInfo:
    """Per-track aggregated info, filled in the Aggregate stage."""

    track_id: int
    cls_id: int                                          # majority class
    cls_name: str
    team_id: Optional[int] = None                        # 0/1 for players, None for ref/ball
    team_label: Optional[str] = None                     # "team_a"/"team_b"/"referee"/"goalkeeper_a"/...
    jersey_number: Optional[int] = None
    jersey_confidence: Optional[float] = None
    n_frames_visible_gate: int = 0
    color_hsv_mean: Optional[Tuple[float, float, float]] = None  # used for team clustering
    first_frame: int = -1
    last_frame: int = -1
    # Mean L2-normalised DINOv3 embedding over visible frames; populated by
    # the embedder pipeline before track merging. ndarray (D,) or None.
    embedding_mean: Optional[Any] = None
    n_observations: int = 0
    # Per-track jersey OCR votes, kept raw so fragments of the same identity
    # can be pooled across a merge before a number is committed.
    #   jersey_votes:       {number -> accumulated weighted vote}
    #   jersey_vote_counts: {number -> n frames that voted for it}
    jersey_votes: Optional[Dict[int, float]] = None
    jersey_vote_counts: Optional[Dict[int, int]] = None


# ------------- Top-level state -------------

@dataclass
class ClipMeta:
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    size_mb: float
    codec: str = ""


@dataclass
class ClipState:
    meta: ClipMeta
    frames: List[FrameInfo] = field(default_factory=list)
    """Indexed by frame number 0..frame_count-1 (sparse if we skip frames)."""
    observations: List[FrameObservation] = field(default_factory=list)
    """Flat list across all frames; group by .track_id for per-track logic."""
    tracks: Dict[int, TrackInfo] = field(default_factory=dict)
    """Filled by aggregate stage."""

    # ------------- Helpers -------------

    def observations_by_track(self) -> Dict[int, List[FrameObservation]]:
        out: Dict[int, List[FrameObservation]] = {}
        for obs in self.observations:
            if obs.track_id is None:
                continue
            out.setdefault(obs.track_id, []).append(obs)
        return out

    def observations_by_frame(self) -> Dict[int, List[FrameObservation]]:
        out: Dict[int, List[FrameObservation]] = {}
        for obs in self.observations:
            out.setdefault(obs.frame_idx, []).append(obs)
        return out

    def to_gsr_json(self) -> Dict[str, Any]:
        """Serialize for /jobs/{id}/gsr.json. Numpy-safe."""

        def jsonable(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o

        meta = asdict(self.meta)
        frames = []
        for f in self.frames:
            frames.append({
                "frame_idx": f.frame_idx,
                "width": f.width,
                "height": f.height,
                "n_keypoints": len(f.keypoints),
                "n_lines": len(f.lines),
                "has_calibration": f.homography_world_to_image is not None,
                "cam_params": f.cam_params,
                "H_world2img": f.homography_world_to_image,
                "H_img2world": f.homography_image_to_world,
            })

        observations = [
            {
                "frame_idx": obs.frame_idx,
                "track_id": obs.track_id,
                "cls_id": obs.cls_id,
                "det_confidence": obs.det_confidence,
                "bbox_xyxy": list(obs.bbox_xyxy),
                "visibility_p": obs.visibility_p,
                "team_id": obs.team_id,
                "pitch_xy": list(obs.pitch_xy) if obs.pitch_xy else None,
            }
            for obs in self.observations
        ]

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
            "meta": jsonable(meta),
            "frames": frames,
            "observations": observations,
            "tracks": tracks,
        }
