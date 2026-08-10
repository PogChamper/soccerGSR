from __future__ import annotations

from app.services.clip_state import ClipMeta, ClipState, FrameObservation
from app.services.video_processor import VideoProcessor


def test_aggregate_estimates_team_and_pitch_before_merge(monkeypatch) -> None:
    events: list[str] = []
    state = ClipState(
        meta=ClipMeta(
            filename="clip.mp4",
            width=100,
            height=100,
            fps=25.0,
            frame_count=1,
            duration=0.04,
            size_mb=0.1,
        ),
        observations=[
            FrameObservation(
                frame_idx=0,
                bbox_xyxy=(0.0, 0.0, 10.0, 20.0),
                cls_id=0,
                det_confidence=0.9,
                track_id=5,
            )
        ],
    )

    class Jersey:
        def collect_votes_into_tracks(self, clip_state):
            events.append("collect")

        def commit_numbers(self, clip_state):
            events.append("commit")

        def dedup_numbers(self, clip_state):
            events.append("dedup")

    class Calibrator:
        def calibrate(self, clip_state):
            events.append("calibrate")
            clip_state.observations[0].pitch_xy = (0.0, 0.0)

    class Teams:
        def fit_and_assign(self, clip_state):
            events.append("team")
            for track in clip_state.tracks.values():
                track.team_id = 0
            for observation in clip_state.observations:
                observation.team_id = 0

        def mean_embedding(self, track_id):
            return None

        def remap_tracks(self, mapping):
            events.append("remap")

    def merge(clip_state, *, embedder_source=None):
        events.append("merge")
        assert clip_state.tracks[5].team_id == 0
        assert clip_state.observations[0].pitch_xy == (0.0, 0.0)
        return {5: 5}

    monkeypatch.setattr("app.services.track_merger.merge_tracks", merge)
    monkeypatch.setattr("app.services.track_smoother.interpolate_track_gaps", lambda *a, **k: 0)

    processor = VideoProcessor(detector=object(), jersey_recognizer=Jersey())
    processor.aggregate(state, team_classifier=Teams(), calibrator=Calibrator())

    assert events == [
        "collect",
        "calibrate",
        "team",
        "merge",
        "remap",
        "commit",
        "team",
        "dedup",
    ]
