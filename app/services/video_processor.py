"""Offline video processing and result rendering."""

from __future__ import annotations

import logging
import math
import os
import time
from collections import Counter
from collections.abc import Callable

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


DEFAULT_FPS = 25.0
ProgressCallback = Callable[[str, float], None]


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_metadata(video_path: str) -> ClipMeta:
    """Read validated container metadata."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        width_value = _finite_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = _finite_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width_value is None or height_value is None:
            raise ValueError(f"Invalid video dimensions: {video_path}")

        width = int(width_value)
        height = int(height_value)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {video_path}")

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_value = _finite_float(raw_fps)
        if fps_value is None or fps_value <= 0:
            logger.warning(
                "invalid video FPS for %s (%r); using %.1f",
                video_path,
                raw_fps,
                DEFAULT_FPS,
            )
            fps = DEFAULT_FPS
        else:
            fps = fps_value

        frame_count_value = _finite_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = max(0, int(frame_count_value)) if frame_count_value is not None else 0
        fourcc_value = _finite_float(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = max(0, int(fourcc_value)) if fourcc_value is not None else 0
    finally:
        cap.release()

    duration = frame_count / fps
    size_bytes = os.path.getsize(video_path)
    codec_bytes = bytes((fourcc >> (8 * index)) & 0xFF for index in range(4))
    codec = codec_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
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


def _foot_xy(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def _class_name(cls_id: int) -> str:
    return settings.class_names.get(cls_id, "unknown")


class _FFmpegSink:
    """Write BGR frames to H.264 through an ffmpeg pipe."""

    def __init__(self, path, fps, width, height):
        import subprocess

        fps_value = _finite_float(fps)
        fps_str = f"{fps_value if fps_value and fps_value > 0 else DEFAULT_FPS:.6g}"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps_str,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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
    """Create an H.264 sink, falling back to OpenCV MP4V."""
    import shutil

    if shutil.which("ffmpeg"):
        try:
            sink = _FFmpegSink(path, fps, width, height)
            logger.info("pass2: writing with ffmpeg/libx264 (crf 18)")
            return sink
        except Exception as exc:
            logger.warning("pass2: ffmpeg sink failed (%s); using cv2 mp4v", exc)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return None
    logger.info("pass2: writing with cv2 mp4v")
    return _Cv2Sink(writer)


def majority_class(observations: list[FrameObservation]) -> int:
    """Return the temporal majority detector class for a track.

    This is the single canonical role used by team assignment, rendering, and
    JSON output. A deterministic class-id tie break keeps runs stable.
    """
    votes = Counter(observation.cls_id for observation in observations)
    return min(votes, key=lambda class_id: (-votes[class_id], class_id)) if votes else 0


class VideoProcessor:
    """Run frame inference, clip aggregation, and result rendering."""

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
        os.makedirs(settings.artifact_dir, exist_ok=True)

    def pass1(
        self,
        input_path: str,
        state: ClipState,
        progress_cb: ProgressCallback | None = None,
        *,
        tracker=None,
        team_classifier=None,
    ) -> None:
        """Collect frame observations and calibration evidence."""
        if tracker is None and self.tracker_factory is not None:
            tracker = self.tracker_factory(state.meta.fps)
        cap = cv2.VideoCapture(input_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_path}")

            frame_idx = 0
            last_log = time.time()
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

                # BoT-SORT expects one feature row per detection.
                embeddings = None
                if self.embedder is not None and detections:
                    bboxes = [tuple(d.bbox) for d in detections]
                    embeddings = self.embedder.embed_boxes(frame, bboxes)

                if tracker is not None:
                    tracked = tracker.update(detections, frame, embeddings=embeddings)
                else:
                    tracked = [(det, None) for det in detections]

                jersey_results = {}
                if self.jersey_recognizer is not None:
                    jersey_indices = [
                        index
                        for index, (detection, track_id) in enumerate(tracked)
                        if detection.class_id in (0, 1) and track_id is not None
                    ]
                    if jersey_indices:
                        batch_results = self.jersey_recognizer.process_boxes(
                            frame,
                            [tracked[index][0].bbox for index in jersey_indices],
                        )
                        jersey_results = dict(zip(jersey_indices, batch_results))

                for det_idx, (det, track_id) in enumerate(tracked):
                    obs = FrameObservation(
                        frame_idx=frame_idx,
                        bbox_xyxy=tuple(det.bbox),
                        cls_id=det.class_id,
                        det_confidence=det.confidence,
                        track_id=track_id,
                        foot_xy_image=_foot_xy(det.bbox),
                    )

                    if det_idx in jersey_results:
                        jr = jersey_results[det_idx]
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
                        team_classifier.observe(track_id, emb)

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
                        100.0 * frame_idx / state.meta.frame_count if state.meta.frame_count else 0
                    )
                    logger.info(
                        "pass1: %d/%d (%.1f%%) observations=%d",
                        frame_idx,
                        state.meta.frame_count,
                        pct,
                        len(state.observations),
                    )
                    last_log = now
                    if progress_cb is not None:
                        progress_cb("pass1", pct)
        finally:
            cap.release()

        if state.meta.frame_count != frame_idx:
            logger.warning(
                "container frame count corrected: reported=%d decoded=%d",
                state.meta.frame_count,
                frame_idx,
            )
            state.meta.frame_count = frame_idx
            state.meta.duration = frame_idx / state.meta.fps

        if progress_cb is not None:
            progress_cb("pass1", 100.0)

    def aggregate(
        self,
        state: ClipState,
        progress_cb: ProgressCallback | None = None,
        *,
        team_classifier=None,
        calibrator=None,
    ) -> None:
        """Aggregate raw observations into calibrated, merged identities."""

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
                    1 for o in obs_list if o.visibility_p is not None and o.visibility_p > 0.5
                ),
            )

        # A provisional number is useful as a high-precision merge anchor. The
        # final number is committed from pooled evidence after merging.
        if self.jersey_recognizer is not None:
            self.jersey_recognizer.collect_votes_into_tracks(state)

        # Offline merging needs team and pitch-space motion. Both are estimated
        # on raw fragments first, then recomputed on the merged identities.
        if calibrator is not None:
            calibrator.calibrate(state)

        if team_classifier is not None:
            team_classifier.fit_and_assign(state)

        from app.services.track_merger import merge_tracks

        n_before = len(state.tracks)
        mapping = merge_tracks(state, embedder_source=team_classifier)
        if mapping:
            if team_classifier is not None:
                team_classifier.remap_tracks(mapping)
            logger.info(
                "track merger: %d -> %d identities",
                n_before,
                len(state.tracks),
            )

        if self.jersey_recognizer is not None:
            self.jersey_recognizer.commit_numbers(state)

        if team_classifier is not None:
            team_classifier.fit_and_assign(state)

        if self.jersey_recognizer is not None:
            self.jersey_recognizer.dedup_numbers(state)

        if calibrator is not None:
            from app.services.track_smoother import (
                filter_ball_outliers,
                interpolate_track_gaps,
                smooth_pitch_xy,
            )

            filter_ball_outliers(state)
            smooth_pitch_xy(state)
            interpolate_track_gaps(state, max_frame_delta=10)

        if progress_cb is not None:
            progress_cb("aggregate", 100.0)

    def pass2(
        self,
        input_path: str,
        output_path: str,
        state: ClipState,
        *,
        draw_legend_flag: bool = True,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        """Render annotated video using state filled by pass1+aggregate."""
        if self.minimap_renderer is not None:
            self.minimap_renderer.reset_clip_state()
        cap = cv2.VideoCapture(input_path)
        processing_failed = False
        sink = None

        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {input_path}")

            obs_by_frame = state.observations_by_frame()
            known_frames = {frame.frame_idx for frame in state.frames}
            last_log = time.time()
            sink = _make_video_sink(
                output_path,
                state.meta.fps,
                state.meta.width,
                state.meta.height,
            )
            if sink is None:
                raise ValueError(f"Cannot create output video: {output_path}")

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                obs_list = obs_by_frame.get(frame_idx, [])
                # Interpolated observations are only rendered on the minimap.
                real_obs = [o for o in obs_list if not o.synthetic]
                detections = []
                for obs in real_obs:
                    tr = state.tracks.get(obs.track_id)
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
                                if obs.track_id in state.tracks
                                else None
                            ),
                        }
                        for obs in real_obs
                    ],
                )

                if draw_legend_flag:
                    annotated = draw_legend(annotated)

                if self.minimap_renderer is not None and frame_idx in known_frames:
                    annotated = self.minimap_renderer.overlay(annotated, obs_list, state.tracks)

                sink.write(annotated)
                frame_idx += 1

                now = time.time()
                if now - last_log > 5.0:
                    pct = (
                        100.0 * frame_idx / state.meta.frame_count if state.meta.frame_count else 0
                    )
                    logger.info(
                        "pass2: %d/%d (%.1f%%)",
                        frame_idx,
                        state.meta.frame_count,
                        pct,
                    )
                    last_log = now
                    if progress_cb is not None:
                        progress_cb("pass2", pct)

            if frame_idx != state.meta.frame_count:
                raise RuntimeError(
                    f"render decoded {frame_idx} frames; expected {state.meta.frame_count}"
                )
        except BaseException:
            processing_failed = True
            raise
        finally:
            cleanup_error = None
            try:
                cap.release()
            except Exception as exc:
                if processing_failed:
                    logger.exception("video capture cleanup failed after processing error")
                else:
                    cleanup_error = exc

            if sink is not None:
                try:
                    sink.release()
                except Exception as exc:
                    if processing_failed:
                        logger.exception("video sink cleanup failed after processing error")
                    elif cleanup_error is not None:
                        logger.exception("video sink cleanup failed after capture cleanup error")
                    else:
                        cleanup_error = exc

            if cleanup_error is not None:
                raise cleanup_error

        if progress_cb is not None:
            progress_cb("pass2", 100.0)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        *,
        source_filename: str | None = None,
        draw_legend_flag: bool = True,
        progress_cb: ProgressCallback | None = None,
    ) -> tuple[str, ClipState]:
        """Run both passes and clip-level aggregation."""
        meta = read_metadata(input_path)
        if source_filename:
            meta.filename = source_filename
        state = ClipState(meta=meta)

        logger.info(
            "pipeline start: file=%s frames=%d duration=%.1fs resolution=%dx%d",
            meta.filename,
            meta.frame_count,
            meta.duration,
            meta.width,
            meta.height,
        )

        tracker = self.tracker_factory(meta.fps) if self.tracker_factory else None
        team_classifier = self.team_classifier_factory() if self.team_classifier_factory else None
        calibrator = (
            self.calibrator_factory(meta.width, meta.height) if self.calibrator_factory else None
        )

        started = time.perf_counter()
        stage_started = time.perf_counter()
        self.pass1(
            input_path,
            state,
            progress_cb,
            tracker=tracker,
            team_classifier=team_classifier,
        )
        pass1_time = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        self.aggregate(
            state,
            progress_cb,
            team_classifier=team_classifier,
            calibrator=calibrator,
        )
        aggregate_time = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        self.pass2(
            input_path,
            output_path,
            state,
            draw_legend_flag=draw_legend_flag,
            progress_cb=progress_cb,
        )
        pass2_time = time.perf_counter() - stage_started

        jersey_tracks = sum(1 for track in state.tracks.values() if track.jersey_number is not None)
        calibrated_frames = sum(1 for f in state.frames if f.homography_world_to_image is not None)
        logger.info(
            "pipeline complete: total=%.1fs pass1=%.1fs aggregate=%.1fs pass2=%.1fs "
            "frames=%d observations=%d tracks=%d jerseys=%d calibrated=%d",
            time.perf_counter() - started,
            pass1_time,
            aggregate_time,
            pass2_time,
            len(state.frames),
            len(state.observations),
            len(state.tracks),
            jersey_tracks,
            calibrated_frames,
        )

        return output_path, state
