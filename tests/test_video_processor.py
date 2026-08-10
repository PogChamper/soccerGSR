from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services import video_processor
from app.services.clip_state import ClipMeta, ClipState
from app.services.video_processor import DEFAULT_FPS, VideoProcessor, read_metadata


class FakeCapture:
    def __init__(self, *, properties=None, frames=(), opened: bool = True):
        self.properties = properties or {}
        self.frames = iter(frames)
        self.opened = opened
        self.release_count = 0

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int):
        return self.properties.get(property_id, 0.0)

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.release_count += 1


def _state() -> ClipState:
    return ClipState(
        meta=ClipMeta(
            filename="clip.mp4",
            width=16,
            height=8,
            fps=25.0,
            frame_count=1,
            duration=0.04,
            size_mb=0.1,
        )
    )


def test_read_metadata_uses_fps_fallback_and_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    capture = FakeCapture(
        properties={
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: float("nan"),
            cv2.CAP_PROP_FRAME_COUNT: 50.0,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"mp4v"),
        }
    )
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)

    metadata = read_metadata(str(path))

    assert metadata.fps == DEFAULT_FPS
    assert metadata.duration == 2.0
    assert metadata.codec == "mp4v"
    assert capture.release_count == 1


def test_read_metadata_rejects_invalid_dimensions_and_releases_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(
        properties={
            cv2.CAP_PROP_FRAME_WIDTH: 0.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
        }
    )
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(ValueError, match="Invalid video dimensions"):
        read_metadata("clip.mp4")

    assert capture.release_count == 1


def test_pass1_releases_capture_after_inference_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(frames=[np.zeros((8, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)

    class Detector:
        def detect(self, frame):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        VideoProcessor(detector=Detector()).pass1("clip.mp4", _state())

    assert capture.release_count == 1


def test_pass1_corrects_reported_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture(frames=[np.zeros((8, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)

    class Detector:
        def detect(self, frame):
            return []

    state = _state()
    state.meta.frame_count = 5
    state.meta.duration = 0.2

    VideoProcessor(detector=Detector()).pass1("clip.mp4", state)

    assert state.meta.frame_count == 1
    assert state.meta.duration == 0.04


def test_pass2_preserves_render_error_when_sink_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(frames=[np.zeros((8, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(video_processor, "draw_detections", lambda frame, *args, **kwargs: frame)

    class Sink:
        def __init__(self):
            self.release_count = 0

        def write(self, frame) -> None:
            raise RuntimeError("render failed")

        def release(self) -> None:
            self.release_count += 1
            raise RuntimeError("cleanup failed")

    sink = Sink()
    monkeypatch.setattr(video_processor, "_make_video_sink", lambda *args: sink)

    with pytest.raises(RuntimeError, match="render failed"):
        VideoProcessor(detector=object()).pass2(
            "clip.mp4",
            "output.mp4",
            _state(),
            draw_legend_flag=False,
        )

    assert capture.release_count == 1
    assert sink.release_count == 1


def test_pass2_releases_capture_when_sink_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture()
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)

    def fail_sink_creation(*args):
        raise RuntimeError("sink creation failed")

    monkeypatch.setattr(video_processor, "_make_video_sink", fail_sink_creation)

    with pytest.raises(RuntimeError, match="sink creation failed"):
        VideoProcessor(detector=object()).pass2("clip.mp4", "output.mp4", _state())

    assert capture.release_count == 1


def test_pass2_rejects_decode_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture(frames=[np.zeros((8, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(video_processor.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(video_processor, "draw_detections", lambda frame, *args, **kwargs: frame)

    class Sink:
        def write(self, frame) -> None:
            pass

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_processor, "_make_video_sink", lambda *args: Sink())
    state = _state()
    state.meta.frame_count = 2

    with pytest.raises(RuntimeError, match="render decoded 1 frames; expected 2"):
        VideoProcessor(detector=object()).pass2(
            "clip.mp4",
            "output.mp4",
            state,
            draw_legend_flag=False,
        )

    assert capture.release_count == 1


def test_process_video_uses_public_source_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = _state().meta
    metadata.filename = "input_internal.mp4"
    monkeypatch.setattr(video_processor, "read_metadata", lambda _path: metadata)
    monkeypatch.setattr(VideoProcessor, "pass1", lambda *args, **kwargs: None)
    monkeypatch.setattr(VideoProcessor, "aggregate", lambda *args, **kwargs: None)
    monkeypatch.setattr(VideoProcessor, "pass2", lambda *args, **kwargs: None)

    _, state = VideoProcessor(detector=object()).process_video(
        "input_internal.mp4",
        "output.mp4",
        source_filename="nott_32m02-32m44.mp4",
    )

    assert state.meta.filename == "nott_32m02-32m44.mp4"
