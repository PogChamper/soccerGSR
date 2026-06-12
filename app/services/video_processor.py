"""Two-pass GSR video processor.

Pass1     — per-frame raw observations (detections, tracks, jersey logits,
            keypoints/lines) into ClipState. GPU-bound.
Aggregate — per-clip post-processing (merge, jersey, calibration, teams,
            smoothing).
Pass2     — render annotated mp4 (bbox + track_id + jersey + minimap).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from app.config import get_settings
from app.services.clip_state import (
    ClipMeta,
    ClipState,
    FrameInfo,
    FrameObservation,
    TrackInfo,
)
from app.services.detector import Detection
from app.utils.visualizer import draw_detections, draw_legend

logger = logging.getLogger(__name__)
settings = get_settings()


ProgressCb = Callable[[str, float], None]


@dataclass
class ProcessingStats:
    processing_time: float = 0.0
    pass1_time: float = 0.0
    aggregate_time: float = 0.0
    pass2_time: float = 0.0
    frames_processed: int = 0
    total_detections: int = 0
    players_count: int = 0
    goalkeepers_count: int = 0
    referees_count: int = 0
    balls_count: int = 0
    n_tracks: int = 0
    n_tracks_with_jersey: int = 0
    n_frames_calibrated: int = 0


def read_metadata(video_path: str) -> ClipMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    size_bytes = os.path.getsize(video_path)
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    return ClipMeta(
        filename=os.path.basename(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        size_mb=size_bytes / (1024 * 1024),
        codec=codec,
    )


def _foot_xy(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def _class_name(cls_id: int) -> str:
    return settings.class_names.get(cls_id, "unknown")


class _FFmpegSink:
    """Write BGR frames to H.264 via an ffmpeg pipe (crisp, small files)."""

    def __init__(self, path, fps, width, height):
        import subprocess

        fps_str = f"{float(fps):.6g}" if fps else "25"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", fps_str, "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def _stderr_tail(self) -> str:
        try:
            err = self._proc.stderr.read() if self._proc.stderr else b""
            return err.decode(errors="replace")[-500:] or "<no stderr>"
        except Exception:
            return "<no stderr>"

    def write(self, frame) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self._proc.wait()
            raise RuntimeError(f"ffmpeg died mid-write: {self._stderr_tail()}") from None

    def release(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code}: {self._stderr_tail()}")


class _Cv2Sink:
    def __init__(self, writer):
        self._w = writer

    def write(self, frame) -> None:
        self._w.write(frame)

    def release(self) -> None:
        self._w.release()


def _make_video_sink(path, fps, width, height):
    """Return a video sink (ffmpeg H.264 if available, else cv2 mp4v)."""
    import shutil

    if shutil.which("ffmpeg"):
        try:
            sink = _FFmpegSink(path, fps, width, height)
            logger.info("pass2: writing with ffmpeg/libx264 (crf 18)")
            return sink
        except Exception as exc:
            logger.warning(f"pass2: ffmpeg sink failed ({exc}); using cv2 mp4v")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        return None
    logger.info("pass2: writing with cv2 mp4v")
    return _Cv2Sink(writer)


def majority_class(obs_list) -> int:
    """Confidence-weighted majority class for a track.

    Using a per-frame argmax for the displayed class makes the label/shape
    flicker (e.g. referee<->player on a referee in black, or a purple GK read
    as referee on some frames). Voting over the whole track with detection
    confidence as weight gives one stable, more confident class per identity.
    """
    votes: dict = {}
    for o in obs_list:
        w = (o.det_confidence or 0.0) + 0.05
        votes[o.cls_id] = votes.get(o.cls_id, 0.0) + w
    return max(votes, key=votes.get) if votes else 0


class VideoProcessor:
    """Two-pass orchestrator. Per-clip stateful phases come in as factories
    (tracker, team classifier, calibrator), shared stateless ones as
    instances. ``tracker_factory`` takes the clip fps (BoT-SORT scales its
    lost-track buffer by frame rate). Any phase may be None to disable it.
    """

    def __init__(
        self,
        detector,
        *,
        tracker_factory=None,
        jersey_recognizer=None,
        team_classifier_factory=None,
        keypoints_extractor=None,
        calibrator_factory=None,
        embedder=None,
        minimap_renderer=None,
    ):
        if detector is None:
            raise ValueError("Detector not initialized")
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.jersey_recognizer = jersey_recognizer
        self.team_classifier_factory = team_classifier_factory
        self.keypoints_extractor = keypoints_extractor
        self.calibrator_factory = calibrator_factory
        self.embedder = embedder
        self.minimap_renderer = minimap_renderer
        os.makedirs(settings.temp_dir, exist_ok=True)

    # ------------------------------------------------------------------ pass 1

    def pass1(
        self,
        input_path: str,
        state: ClipState,
        progress_cb: Optional[ProgressCb] = None,
        *,
        tracker=None,
        team_classifier=None,
    ) -> None:
        """Iterate frames, fill state.observations / state.frames raw fields."""
        if tracker is None and self.tracker_factory is not None:
            tracker = self.tracker_factory(state.meta.fps)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        frame_idx = 0
        last_log = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_info = FrameInfo(
                    frame_idx=frame_idx,
                    width=state.meta.width,
                    height=state.meta.height,
                )

                detections = self.detector.detect(frame)

                # One batched embedding pass per frame, all classes: ball/ref
                # crops are overkill but cheap, and BoT-SORT's appearance
                # gate then works uniformly.
                embeddings = None
                if self.embedder is not None and detections:
                    bboxes = [tuple(d.bbox) for d in detections]
                    embeddings = self.embedder.embed_boxes(frame, bboxes)

                if tracker is not None:
                    tracked = tracker.update(detections, frame, embeddings=embeddings)
                else:
                    tracked = [(det, None) for det in detections]

                for det_idx, (det, track_id) in enumerate(tracked):
                    obs = FrameObservation(
                        frame_idx=frame_idx,
                        bbox_xyxy=tuple(det.bbox),
                        cls_id=det.class_id,
                        det_confidence=det.confidence,
                        track_id=track_id,
                        foot_xy_image=_foot_xy(det.bbox),
                    )

                    if (
                        self.jersey_recognizer is not None
                        and det.class_id in (0, 1)        # players + goalkeepers
                        and track_id is not None
                    ):
                        jr = self.jersey_recognizer.process_crop(frame, det.bbox)
                        obs.visibility_p = jr.visibility_p
                        obs.ocr_logits_tens = jr.ocr_logits_tens
                        obs.ocr_logits_units = jr.ocr_logits_units

                    if (
                        team_classifier is not None
                        and det.class_id in (0, 1)
                        and track_id is not None
                    ):
                        emb = (
                            embeddings[det_idx]
                            if embeddings is not None and det_idx < len(embeddings)
                            else None
                        )
                        team_classifier.observe(
                            track_id, emb, det.class_id, frame=frame, bbox=det.bbox
                        )

                    state.observations.append(obs)

                if self.keypoints_extractor is not None:
                    kp_dict, lines_dict = self.keypoints_extractor.extract(frame)
                    frame_info.keypoints = kp_dict
                    frame_info.lines = lines_dict

                state.frames.append(frame_info)
                frame_idx += 1

                now = time.time()
                if now - last_log > 5.0:
                    pct = (
                        100.0 * frame_idx / state.meta.frame_count
                        if state.meta.frame_count
                        else 0
                    )
                    logger.info(
                        f"pass1: {frame_idx}/{state.meta.frame_count} "
                        f"({pct:.1f}%) detections={len(state.observations)}"
                    )
                    last_log = now
                    if progress_cb is not None:
                        progress_cb("pass1", pct)
        finally:
            cap.release()

        if progress_cb is not None:
            progress_cb("pass1", 100.0)

    # ----------------------------------------------------------- aggregate

    def aggregate(
        self,
        state: ClipState,
        progress_cb: Optional[ProgressCb] = None,
        *,
        team_classifier=None,
        calibrator=None,
    ) -> None:
        """Build TrackInfo per track, run team clustering / jersey aggregation /
        calibration."""
        from collections import Counter

        by_track = state.observations_by_track()
        for tid, obs_list in by_track.items():
            obs_list.sort(key=lambda o: o.frame_idx)
            cls_majority = majority_class(obs_list)
            state.tracks[tid] = TrackInfo(
                track_id=tid,
                cls_id=cls_majority,
                cls_name=_class_name(cls_majority),
                first_frame=obs_list[0].frame_idx,
                last_frame=obs_list[-1].frame_idx,
                n_frames_visible_gate=sum(
                    1 for o in obs_list
                    if o.visibility_p is not None and o.visibility_p > 0.5
                ),
            )

        # Collect raw jersey votes per fragment; the authoritative number is
        # committed after the merge from each identity's pooled votes.
        if self.jersey_recognizer is not None:
            self.jersey_recognizer.collect_votes_into_tracks(state)

        # Merge fragmented tracks before team clustering (one identity = one
        # set of colour samples); remap_tracks keeps classifier state in sync.
        if team_classifier is not None:
            from app.services.track_merger import merge_tracks

            n_before = len(state.tracks)
            mapping = merge_tracks(state, embedder_source=team_classifier)
            n_after = len(state.tracks)
            if mapping:
                team_classifier.remap_tracks(mapping)
                logger.info(
                    f"track_merger: {n_before} -> {n_after} tracks "
                    f"({n_before - n_after} consolidated)"
                )

        # Now commit jersey numbers on the merged (pooled-vote) identities.
        if self.jersey_recognizer is not None:
            self.jersey_recognizer.commit_numbers(state)

        if calibrator is not None:
            calibrator.calibrate(state)

        if team_classifier is not None:
            team_classifier.fit_and_assign(state)

        # Jersey dedup needs team assignment: numbers are only unique within
        # a team (both teams routinely field a #10).
        if self.jersey_recognizer is not None:
            self.jersey_recognizer.dedup_numbers(state)

        if calibrator is not None:
            from app.services.track_smoother import (
                filter_ball_outliers,
                interpolate_track_gaps,
                smooth_detection_class,
                smooth_pitch_xy,
            )
            filter_ball_outliers(state)
            smooth_pitch_xy(state)
            smooth_detection_class(state)
            # gap-fill last, so synthetic points aren't fed to the smoother
            interpolate_track_gaps(state)

        if progress_cb is not None:
            progress_cb("aggregate", 100.0)

    # ---------------------------------------------------------------- pass 2

    def pass2(
        self,
        input_path: str,
        output_path: str,
        state: ClipState,
        *,
        draw_legend_flag: bool = True,
        progress_cb: Optional[ProgressCb] = None,
    ) -> None:
        """Render annotated video using state filled by pass1+aggregate."""
        if self.minimap_renderer is not None:
            # the renderer is shared across jobs; its snap-cache / frozen-frame
            # state is clip-scoped and must not leak from the previous clip
            self.minimap_renderer.reset_clip_state()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        sink = _make_video_sink(
            output_path, state.meta.fps, state.meta.width, state.meta.height
        )
        if sink is None:
            cap.release()
            raise ValueError(f"Cannot create output video: {output_path}")

        obs_by_frame = state.observations_by_frame()
        last_log = time.time()

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                obs_list = obs_by_frame.get(frame_idx, [])
                # synthetic gap-fill obs are minimap-only; never draw them as
                # fabricated bounding boxes on the main video.
                real_obs = [o for o in obs_list if not o.synthetic]
                detections = []
                for obs in real_obs:
                    tr = state.tracks.get(obs.track_id)
                    # space-time-voted class (track-independent) wins; falls back
                    # to the stable per-track class so labels/colors don't flicker
                    if obs.display_cls is not None:
                        scid = obs.display_cls
                    else:
                        scid = tr.cls_id if tr is not None else obs.cls_id
                    detections.append(
                        Detection(
                            bbox=obs.bbox_xyxy,
                            class_id=scid,
                            class_name=_class_name(scid),
                            confidence=obs.det_confidence,
                        )
                    )
                annotated = draw_detections(
                    frame,
                    detections,
                    extras=[
                        {
                            "track_id": obs.track_id,
                            "team_id": obs.team_id,
                            "jersey_number": (
                                state.tracks[obs.track_id].jersey_number
                                if obs.track_id in state.tracks else None
                            ),
                        }
                        for obs in real_obs
                    ],
                )

                if draw_legend_flag:
                    annotated = draw_legend(annotated)

                if self.minimap_renderer is not None and frame_idx < len(state.frames):
                    annotated = self.minimap_renderer.overlay(
                        annotated,
                        state.frames[frame_idx],
                        obs_list,
                        state.tracks,
                    )

                sink.write(annotated)
                frame_idx += 1

                now = time.time()
                if now - last_log > 5.0:
                    pct = (
                        100.0 * frame_idx / state.meta.frame_count
                        if state.meta.frame_count
                        else 0
                    )
                    logger.info(
                        f"pass2: {frame_idx}/{state.meta.frame_count} ({pct:.1f}%)"
                    )
                    last_log = now
                    if progress_cb is not None:
                        progress_cb("pass2", pct)
        finally:
            sink.release()
            cap.release()

        if progress_cb is not None:
            progress_cb("pass2", 100.0)

    # -------------------------------------------------------------- top-level

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        *,
        draw_legend_flag: bool = True,
        progress_cb: Optional[ProgressCb] = None,
    ) -> Tuple[str, ClipState, ProcessingStats]:
        """Run pass1 -> aggregate -> pass2. Returns (output_path, state, stats)."""
        meta = read_metadata(input_path)
        state = ClipState(meta=meta)
        stats = ProcessingStats()

        if output_path is None:
            output_path = os.path.join(
                settings.temp_dir,
                f"output_{int(time.time())}_{meta.filename}",
            )
            if not output_path.endswith(".mp4"):
                output_path = os.path.splitext(output_path)[0] + ".mp4"

        logger.info(
            f"GSR pipeline start: {meta.filename} "
            f"({meta.frame_count} frames, {meta.duration:.1f}s, {meta.width}x{meta.height})"
        )

        # Per-clip services that need state shared between pass1 and aggregate.
        tracker = self.tracker_factory(meta.fps) if self.tracker_factory else None
        team_classifier = (
            self.team_classifier_factory() if self.team_classifier_factory else None
        )
        calibrator = (
            self.calibrator_factory(meta.width, meta.height)
            if self.calibrator_factory
            else None
        )

        t_total = time.time()
        t = time.time()
        self.pass1(
            input_path,
            state,
            progress_cb,
            tracker=tracker,
            team_classifier=team_classifier,
        )
        stats.pass1_time = time.time() - t

        t = time.time()
        self.aggregate(
            state,
            progress_cb,
            team_classifier=team_classifier,
            calibrator=calibrator,
        )
        stats.aggregate_time = time.time() - t

        t = time.time()
        self.pass2(
            input_path,
            output_path,
            state,
            draw_legend_flag=draw_legend_flag,
            progress_cb=progress_cb,
        )
        stats.pass2_time = time.time() - t

        # ---- summarise stats ----
        stats.frames_processed = len(state.frames)
        stats.total_detections = len(state.observations)
        for obs in state.observations:
            if obs.cls_id == 0:
                stats.players_count += 1
            elif obs.cls_id == 1:
                stats.goalkeepers_count += 1
            elif obs.cls_id == 2:
                stats.referees_count += 1
            elif obs.cls_id == 3:
                stats.balls_count += 1
        stats.n_tracks = len(state.tracks)
        stats.n_tracks_with_jersey = sum(
            1 for t in state.tracks.values() if t.jersey_number is not None
        )
        stats.n_frames_calibrated = sum(
            1 for f in state.frames if f.homography_world_to_image is not None
        )
        stats.processing_time = time.time() - t_total

        logger.info(
            f"GSR pipeline done: {stats.processing_time:.1f}s "
            f"(p1={stats.pass1_time:.1f} agg={stats.aggregate_time:.1f} "
            f"p2={stats.pass2_time:.1f}) "
            f"frames={stats.frames_processed} dets={stats.total_detections} "
            f"tracks={stats.n_tracks} jerseys={stats.n_tracks_with_jersey} "
            f"calibrated={stats.n_frames_calibrated}"
        )

        return output_path, state, stats
