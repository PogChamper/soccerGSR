from __future__ import annotations

import numpy as np

from app.services.clip_state import ClipMeta, ClipState, FrameObservation, TrackInfo
from app.services.team_classifier import TeamClassifier, _left_cluster_by_frame


def _state(
    tracks: dict[int, TrackInfo],
    observations: list[FrameObservation] | None = None,
) -> ClipState:
    return ClipState(
        meta=ClipMeta("clip.mp4", 100, 100, 25.0, 2, 0.08, 0.1),
        tracks=tracks,
        observations=observations or [],
    )


def test_left_cluster_uses_equal_per_frame_votes() -> None:
    state = _state(
        {},
        [
            FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=1, pitch_xy=(-20, 0)),
            FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=2, pitch_xy=(-10, 0)),
            FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=3, pitch_xy=(15, 0)),
            FrameObservation(1, (0, 0, 1, 2), 0, 0.9, track_id=1, pitch_xy=(-5, 0)),
            FrameObservation(1, (0, 0, 1, 2), 0, 0.9, track_id=3, pitch_xy=(10, 0)),
        ],
    )

    assert _left_cluster_by_frame(state, {1: 0, 2: 0, 3: 1}) == 0


def test_left_cluster_falls_back_to_track_positions() -> None:
    state = _state(
        {},
        [
            FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=1, pitch_xy=(-5, 0)),
            FrameObservation(1, (0, 0, 1, 2), 0, 0.9, track_id=2, pitch_xy=(5, 0)),
        ],
    )

    assert _left_cluster_by_frame(state, {1: 0, 2: 1}) == 0


def test_embedding_mean_uses_every_observation() -> None:
    classifier = TeamClassifier()
    for _ in range(60):
        classifier.observe(1, np.array([1.0, 0.0], dtype=np.float32))
    for _ in range(40):
        classifier.observe(1, np.array([0.0, 1.0], dtype=np.float32))

    mean = classifier.mean_embedding(1)

    assert mean is not None
    np.testing.assert_allclose(mean, np.array([0.6, 0.4]) / np.hypot(0.6, 0.4))
    assert classifier._embeddings[1].count == 100


def test_remap_tracks_pools_embedding_sums() -> None:
    classifier = TeamClassifier()
    classifier.observe(1, np.array([1.0, 0.0], dtype=np.float32))
    classifier.observe(2, np.array([0.0, 1.0], dtype=np.float32))

    classifier.remap_tracks({1: 7, 2: 7})

    np.testing.assert_allclose(
        classifier.mean_embedding(7),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
    )
    assert classifier._embeddings[7].count == 2


def test_assigns_teams_from_track_embeddings_and_pitch_orientation() -> None:
    tracks = {
        track_id: TrackInfo(track_id=track_id, cls_id=0, cls_name="player")
        for track_id in range(1, 5)
    }
    observations = [
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=1, pitch_xy=(-20, 0)),
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=2, pitch_xy=(-10, 0)),
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=3, pitch_xy=(10, 0)),
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=4, pitch_xy=(20, 0)),
    ]
    classifier = TeamClassifier()
    for track_id, feature in {
        1: (1.0, 0.0),
        2: (0.95, 0.05),
        3: (-1.0, 0.0),
        4: (-0.95, 0.05),
    }.items():
        classifier.observe(track_id, np.asarray(feature, dtype=np.float32))

    state = _state(tracks, observations)
    classifier.fit_and_assign(state)

    assert [state.tracks[index].team_label for index in (1, 2)] == [
        "team_left",
        "team_left",
    ]
    assert [state.tracks[index].team_label for index in (3, 4)] == [
        "team_right",
        "team_right",
    ]
    assert [observation.team_id for observation in state.observations] == [0, 0, 1, 1]


def test_goalkeeper_side_uses_pitch_position() -> None:
    tracks = {
        1: TrackInfo(track_id=1, cls_id=0, cls_name="player"),
        2: TrackInfo(track_id=2, cls_id=0, cls_name="player"),
        3: TrackInfo(track_id=3, cls_id=1, cls_name="goalkeeper"),
    }
    observations = [
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=1, pitch_xy=(-20, 0)),
        FrameObservation(0, (0, 0, 1, 2), 0, 0.9, track_id=2, pitch_xy=(20, 0)),
        FrameObservation(0, (0, 0, 1, 2), 1, 0.9, track_id=3, pitch_xy=(45, 0)),
    ]
    classifier = TeamClassifier()
    classifier.observe(1, np.array([1.0, 0.0], dtype=np.float32))
    classifier.observe(2, np.array([-1.0, 0.0], dtype=np.float32))
    classifier.observe(3, np.array([1.0, 0.0], dtype=np.float32))

    state = _state(tracks, observations)
    classifier.fit_and_assign(state)

    assert state.tracks[3].team_id == 1
    assert state.tracks[3].team_label == "goalkeeper_right"


def test_unoriented_clusters_use_anonymous_labels() -> None:
    tracks = {
        1: TrackInfo(track_id=1, cls_id=0, cls_name="player"),
        2: TrackInfo(track_id=2, cls_id=0, cls_name="player"),
    }
    classifier = TeamClassifier()
    classifier.observe(1, np.array([1.0, 0.0], dtype=np.float32))
    classifier.observe(2, np.array([-1.0, 0.0], dtype=np.float32))

    state = _state(tracks)
    classifier.fit_and_assign(state)

    assert {track.team_label for track in state.tracks.values()} == {"team_0", "team_1"}


def test_team_assignment_does_not_change_canonical_roles() -> None:
    tracks = {
        1: TrackInfo(track_id=1, cls_id=2, cls_name="referee"),
        2: TrackInfo(track_id=2, cls_id=3, cls_name="ball"),
    }
    classifier = TeamClassifier()
    classifier.observe(1, np.array([1.0, 0.0], dtype=np.float32))

    state = _state(tracks)
    classifier.fit_and_assign(state)

    assert state.tracks[1].team_label == "referee"
    assert state.tracks[2].team_label == "ball"
    assert all(track.team_id is None for track in state.tracks.values())
