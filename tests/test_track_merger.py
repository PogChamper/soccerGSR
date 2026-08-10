from __future__ import annotations

import numpy as np

from app.services.clip_state import ClipMeta, ClipState, FrameObservation, TrackInfo
from app.services.jersey import JerseyRecognizer, JerseyVoteConfig
from app.services.track_merger import MergeConfig, merge_tracks


def _state(specs: list[dict], *, fps: float = 25.0) -> ClipState:
    state = ClipState(
        meta=ClipMeta(
            filename="clip.mp4",
            width=1920,
            height=1080,
            fps=fps,
            frame_count=200,
            duration=8.0,
            size_mb=1.0,
        )
    )
    for spec in specs:
        track_id = spec["track_id"]
        frames = spec["frames"]
        state.tracks[track_id] = TrackInfo(
            track_id=track_id,
            cls_id=spec.get("cls_id", 0),
            cls_name="player",
            team_id=spec.get("team_id"),
            team_label=(f"team_{spec['team_id']}" if spec.get("team_id") is not None else None),
            jersey_number=spec.get("jersey_number"),
            jersey_confidence=spec.get("jersey_confidence", 0.99),
            first_frame=min(frames),
            last_frame=max(frames),
            n_observations=len(frames),
            jersey_logits_tens=spec.get("jersey_logits_tens"),
            jersey_logits_units=spec.get("jersey_logits_units"),
            jersey_vote_count=spec.get("jersey_vote_count", 0),
            embedding_mean=spec.get("embedding"),
        )
        for frame, pitch in zip(frames, spec.get("pitch", [None] * len(frames))):
            state.observations.append(
                FrameObservation(
                    frame_idx=frame,
                    bbox_xyxy=(0.0, 0.0, 10.0, 20.0),
                    cls_id=spec.get("cls_id", 0),
                    det_confidence=0.9,
                    track_id=track_id,
                    pitch_xy=pitch,
                    team_id=spec.get("team_id"),
                )
            )
    return state


def test_same_jersey_on_opposing_teams_is_not_merged() -> None:
    state = _state(
        [
            {"track_id": 10, "frames": [0, 1], "team_id": 0, "jersey_number": 10},
            {"track_id": 20, "frames": [2, 3], "team_id": 1, "jersey_number": 10},
        ]
    )

    mapping = merge_tracks(state)

    assert len(state.tracks) == 2
    assert mapping[10] != mapping[20]


def test_same_team_and_jersey_merges_and_pools_votes() -> None:
    first_tens = [10.0] + [0.0] * 9
    first_units = [0.0] * 8 + [10.0, 0.0]
    second_tens = [20.0] + [0.0] * 9
    second_units = [0.0] * 8 + [20.0, 0.0]
    state = _state(
        [
            {
                "track_id": 4,
                "frames": [0, 1],
                "team_id": 0,
                "jersey_number": 8,
                "jersey_logits_tens": first_tens,
                "jersey_logits_units": first_units,
                "jersey_vote_count": 2,
            },
            {
                "track_id": 9,
                "frames": [20, 21],
                "team_id": 0,
                "jersey_number": 8,
                "jersey_logits_tens": second_tens,
                "jersey_logits_units": second_units,
                "jersey_vote_count": 4,
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[4] == mapping[9]
    assert len(state.tracks) == 1
    merged = next(iter(state.tracks.values()))
    assert merged.jersey_logits_tens == [30.0] + [0.0] * 9
    assert merged.jersey_logits_units == [0.0] * 8 + [30.0, 0.0]
    assert merged.jersey_vote_count == 6

    recognizer = JerseyRecognizer.__new__(JerseyRecognizer)
    recognizer.vote_config = JerseyVoteConfig()
    recognizer.commit_numbers(state)
    assert merged.jersey_number == 8


def test_overlap_invariant_blocks_merge() -> None:
    state = _state(
        [
            {"track_id": 1, "frames": [0, 1], "team_id": 0, "jersey_number": 7},
            {"track_id": 2, "frames": [1, 2], "team_id": 0, "jersey_number": 7},
        ]
    )

    mapping = merge_tracks(state)

    assert len(state.tracks) == 2
    assert mapping[1] != mapping[2]


def test_motion_merge_cannot_create_transitive_jersey_conflict() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "jersey_number": 7,
                "pitch": [(0.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [1],
                "team_id": 0,
                "pitch": [(0.2, 0.0)],
            },
            {
                "track_id": 3,
                "frames": [2],
                "team_id": 0,
                "jersey_number": 9,
                "pitch": [(0.4, 0.0)],
            },
        ]
    )

    merge_tracks(state)

    assert len(state.tracks) == 2
    assert {track.jersey_number for track in state.tracks.values()} == {7, 9}


def test_feasible_pitch_motion_merges_fragments() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0, 1],
                "team_id": 0,
                "pitch": [(0.0, 0.0), (1.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [26, 27],
                "team_id": 0,
                "pitch": [(8.0, 0.0), (9.0, 0.0)],
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[1] == mapping[2]


def test_motion_uses_the_tail_of_a_merged_component() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "pitch": [(0.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [25],
                "team_id": 0,
                "pitch": [(9.0, 0.0)],
            },
            {
                "track_id": 3,
                "frames": [26],
                "team_id": 0,
                "pitch": [(0.0, 0.0)],
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[1] == mapping[2]
    assert mapping[3] != mapping[2]


def test_motion_tries_a_disjoint_candidate_when_closest_component_overlaps() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "jersey_number": 7,
                "pitch": [(0.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [10],
                "team_id": 0,
                "jersey_number": 7,
                "pitch": [(40.0, 0.0)],
            },
            {
                "track_id": 4,
                "frames": [1],
                "team_id": 0,
                "pitch": [(1.0, 0.0)],
            },
            {
                "track_id": 5,
                "frames": [10],
                "team_id": 0,
                "pitch": [(0.1, 0.0)],
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[4] == mapping[5]
    assert mapping[1] != mapping[5]


def test_impossible_pitch_motion_and_short_tracks_are_preserved() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "pitch": [(0.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [2],
                "team_id": 0,
                "pitch": [(50.0, 0.0)],
            },
        ]
    )

    mapping = merge_tracks(
        state,
        config=MergeConfig(max_gap_seconds=1.0, max_speed_mps=9.0, distance_slack_m=1.0),
    )

    assert len(state.tracks) == 2
    assert mapping[1] != mapping[2]
    assert all(observation.track_id is not None for observation in state.observations)


def test_low_confidence_jersey_does_not_anchor_a_merge() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0, 1],
                "team_id": 0,
                "jersey_number": 7,
                "jersey_confidence": 0.3,
            },
            {
                "track_id": 2,
                "frames": [10, 11],
                "team_id": 0,
                "jersey_number": 7,
                "jersey_confidence": 0.3,
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[1] != mapping[2]


def test_impossible_motion_blocks_an_appearance_merge() -> None:
    state = _state(
        [
            {
                "track_id": 1,
                "frames": [0, 1],
                "team_id": 0,
                "embedding": np.array([1, 0]),
                "pitch": [(-50.0, 0.0), (-50.0, 0.0)],
            },
            {
                "track_id": 2,
                "frames": [4, 5],
                "team_id": 0,
                "embedding": np.array([1, 0]),
                "pitch": [(50.0, 0.0), (50.0, 0.0)],
            },
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[1] != mapping[2]


def test_osnet_similarity_merges_disjoint_same_team_fragments() -> None:
    second = np.array([0.95, np.sqrt(1.0 - 0.95**2)], dtype=np.float32)
    state = _state(
        [
            {"track_id": 1, "frames": [0], "team_id": 0, "embedding": np.array([1, 0])},
            {"track_id": 2, "frames": [100], "team_id": 0, "embedding": second},
        ]
    )

    mapping = merge_tracks(state)

    assert mapping[1] == mapping[2]


def test_osnet_similarity_respects_threshold_and_jersey_conflicts() -> None:
    below_threshold = np.array([0.89, np.sqrt(1.0 - 0.89**2)], dtype=np.float32)
    threshold_state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "embedding": np.array([1, 0]),
            },
            {
                "track_id": 2,
                "frames": [100],
                "team_id": 0,
                "embedding": below_threshold,
            },
        ]
    )
    conflict_state = _state(
        [
            {
                "track_id": 1,
                "frames": [0],
                "team_id": 0,
                "jersey_number": 7,
                "embedding": np.array([1, 0]),
            },
            {
                "track_id": 2,
                "frames": [100],
                "team_id": 0,
                "jersey_number": 9,
                "embedding": np.array([1, 0]),
            },
        ]
    )

    threshold_mapping = merge_tracks(threshold_state)
    conflict_mapping = merge_tracks(conflict_state)

    assert threshold_mapping[1] != threshold_mapping[2]
    assert conflict_mapping[1] != conflict_mapping[2]
