"""Offline consolidation of tracker fragments into stable identities.

The merger only uses clip-level evidence available at inference time. Person
fragments are linked by high-confidence ``(team, jersey)`` agreement, feasible
motion in pitch coordinates, and optionally a validated appearance threshold.
Every union preserves the invariant of at most one observation per identity and
frame.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.services.clip_state import ClipState, TrackInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeConfig:
    """Thresholds validated with the SoccerNet OSNet pipeline."""

    max_gap_seconds: float = 6.0
    max_speed_mps: float = 9.0
    distance_slack_m: float = 7.0
    min_cosine_similarity: float | None = 0.90
    min_jersey_confidence: float = 0.65


class EmbeddingSource(Protocol):
    def mean_embedding(self, track_id: int) -> np.ndarray | None: ...


@dataclass
class _Fragment:
    track_id: int
    cls_id: int
    team_id: int | None
    jersey_number: int | None
    jersey_confidence: float | None
    frames: set[int]
    first_frame: int
    last_frame: int
    first_pitch_frame: int | None
    last_pitch_frame: int | None
    first_pitch: tuple[float, float] | None
    last_pitch: tuple[float, float] | None
    embedding: np.ndarray | None


class _OverlapAwareUnionFind:
    """Union-find that preserves frame and identity consistency."""

    def __init__(self, fragments: dict[int, _Fragment]) -> None:
        self.parent = {track_id: track_id for track_id in fragments}
        self.frames = {track_id: set(fragment.frames) for track_id, fragment in fragments.items()}
        self.jerseys = {
            track_id: ({fragment.jersey_number} if fragment.jersey_number is not None else set())
            for track_id, fragment in fragments.items()
        }
        self.members = {track_id: {track_id} for track_id in fragments}

    def find(self, track_id: int) -> int:
        parent = self.parent
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def can_union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if not self.frames[left_root].isdisjoint(self.frames[right_root]):
            return False
        # Unknown numbers can join a numbered component. Two different known
        # numbers cannot become one identity through transitive motion links.
        known_numbers = self.jerseys[left_root] | self.jerseys[right_root]
        if len(known_numbers) > 1:
            return False
        return True

    def union(self, left: int, right: int) -> bool:
        if not self.can_union(left, right):
            return False
        left_root = self.find(left)
        right_root = self.find(right)

        # Stable roots make output deterministic across runs.
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.frames[left_root].update(self.frames.pop(right_root))
        self.jerseys[left_root].update(self.jerseys.pop(right_root))
        self.members[left_root].update(self.members.pop(right_root))
        return True


@dataclass(frozen=True)
class _MotionComponent:
    root: int
    scope: _Fragment
    first_frame: int
    last_frame: int
    first_pitch: tuple[float, float]
    last_pitch: tuple[float, float]


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return (np.asarray(vector, dtype=np.float32) / norm).astype(np.float32)


def _same_scope(left: _Fragment, right: _Fragment) -> bool:
    """Return whether two fragments may represent the same field object."""
    if left.cls_id != right.cls_id:
        return False
    if left.cls_id in (0, 1):
        return (
            left.team_id is not None and right.team_id is not None and left.team_id == right.team_id
        )
    return left.cls_id in (2, 3)


def _jersey_compatible(left: _Fragment, right: _Fragment) -> bool:
    return (
        left.jersey_number is None
        or right.jersey_number is None
        or left.jersey_number == right.jersey_number
    )


def _build_fragments(
    state: ClipState,
    embedding_source: EmbeddingSource | None,
) -> dict[int, _Fragment]:
    observations = state.observations_by_track()
    fragments: dict[int, _Fragment] = {}

    for track_id, track in state.tracks.items():
        track_observations = sorted(observations.get(track_id, ()), key=lambda item: item.frame_idx)
        if not track_observations:
            continue
        pitch = [
            (observation.frame_idx, observation.pitch_xy)
            for observation in track_observations
            if observation.pitch_xy is not None
        ]
        embedding = (
            embedding_source.mean_embedding(track_id)
            if embedding_source is not None
            else _unit(track.embedding_mean)
        )
        fragments[track_id] = _Fragment(
            track_id=track_id,
            cls_id=track.cls_id,
            team_id=track.team_id,
            jersey_number=track.jersey_number,
            jersey_confidence=track.jersey_confidence,
            frames={observation.frame_idx for observation in track_observations},
            first_frame=track_observations[0].frame_idx,
            last_frame=track_observations[-1].frame_idx,
            first_pitch_frame=pitch[0][0] if pitch else None,
            last_pitch_frame=pitch[-1][0] if pitch else None,
            first_pitch=pitch[0][1] if pitch else None,
            last_pitch=pitch[-1][1] if pitch else None,
            embedding=_unit(embedding),
        )
    return fragments


def _merge_by_jersey(
    fragments: dict[int, _Fragment],
    union_find: _OverlapAwareUnionFind,
    config: MergeConfig,
) -> int:
    groups: dict[tuple[int, int], list[_Fragment]] = defaultdict(list)
    for fragment in fragments.values():
        if (
            fragment.cls_id == 0
            and fragment.team_id is not None
            and fragment.jersey_number is not None
            and (fragment.jersey_confidence or 0.0) >= config.min_jersey_confidence
        ):
            groups[(fragment.team_id, fragment.jersey_number)].append(fragment)

    merged = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (item.first_frame, item.track_id))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                merged += int(union_find.union(left.track_id, right.track_id))
    return merged


def _merge_by_motion(
    fragments: dict[int, _Fragment],
    union_find: _OverlapAwareUnionFind,
    config: MergeConfig,
    fps: float,
) -> int:
    def components() -> list[_MotionComponent]:
        result: list[_MotionComponent] = []
        for root, track_ids in sorted(union_find.members.items()):
            members = [fragments[track_id] for track_id in track_ids]
            with_start = [member for member in members if member.first_pitch_frame is not None]
            with_end = [member for member in members if member.last_pitch_frame is not None]
            if not with_start or not with_end:
                continue
            first = min(
                with_start,
                key=lambda member: (member.first_pitch_frame, member.track_id),
            )
            last = max(
                with_end,
                key=lambda member: (member.last_pitch_frame, -member.track_id),
            )
            assert first.first_pitch_frame is not None and first.first_pitch is not None
            assert last.last_pitch_frame is not None and last.last_pitch is not None
            result.append(
                _MotionComponent(
                    root=root,
                    scope=fragments[min(track_ids)],
                    first_frame=int(first.first_pitch_frame),
                    last_frame=int(last.last_pitch_frame),
                    first_pitch=first.first_pitch,
                    last_pitch=last.last_pitch,
                )
            )
        return result

    merged = 0
    max_gap_frames = max(1, int(round(config.max_gap_seconds * fps)))

    # Merge one component at a time and recompute its chronological tail.
    # Otherwise a later fragment could incorrectly link back to an old member
    # of a component and bypass the speed constraint.
    while True:
        values = components()
        merged_this_round = False
        for current in sorted(values, key=lambda item: (item.first_frame, item.root)):
            candidates: list[tuple[float, int, int]] = []
            for previous in values:
                if previous.root == current.root:
                    continue
                if not _same_scope(previous.scope, current.scope):
                    continue
                if not union_find.can_union(previous.root, current.root):
                    continue
                gap = current.first_frame - previous.last_frame
                if gap <= 0 or gap > max_gap_frames:
                    continue
                distance = float(
                    np.hypot(
                        previous.last_pitch[0] - current.first_pitch[0],
                        previous.last_pitch[1] - current.first_pitch[1],
                    )
                )
                reachable = config.max_speed_mps * (gap / fps) + config.distance_slack_m
                if distance <= reachable:
                    # Distance, then temporal gap, then stable root resolve ties.
                    candidates.append((distance, gap, previous.root))

            for _, _, previous_root in sorted(candidates):
                if union_find.union(previous_root, current.root):
                    merged += 1
                    merged_this_round = True
                    break
            if merged_this_round:
                break
        if not merged_this_round:
            return merged


def _motion_feasible(left: _Fragment, right: _Fragment, config: MergeConfig, fps: float) -> bool:
    """Refuse a pair only when known pitch endpoints prove an impossible move."""
    earlier, later = sorted((left, right), key=lambda fragment: fragment.first_frame)
    if (
        earlier.last_pitch is None
        or later.first_pitch is None
        or later.first_pitch_frame <= earlier.last_pitch_frame
    ):
        return True
    gap = later.first_pitch_frame - earlier.last_pitch_frame
    distance = float(
        np.hypot(
            earlier.last_pitch[0] - later.first_pitch[0],
            earlier.last_pitch[1] - later.first_pitch[1],
        )
    )
    return distance <= config.max_speed_mps * (gap / fps) + config.distance_slack_m


def _merge_by_appearance(
    fragments: dict[int, _Fragment],
    union_find: _OverlapAwareUnionFind,
    config: MergeConfig,
    fps: float,
) -> int:
    threshold = config.min_cosine_similarity
    if threshold is None:
        return 0

    candidates: list[tuple[float, int, int]] = []
    values = list(fragments.values())
    for index, left in enumerate(values):
        if left.embedding is None or left.cls_id == 3:
            continue
        for right in values[index + 1 :]:
            if right.embedding is None:
                continue
            if not _same_scope(left, right) or not _jersey_compatible(left, right):
                continue
            if not _motion_feasible(left, right, config, fps):
                continue
            similarity = float(left.embedding @ right.embedding)
            if similarity >= threshold:
                candidates.append((similarity, left.track_id, right.track_id))

    merged = 0
    for _, left, right in sorted(candidates, reverse=True):
        merged += int(union_find.union(left, right))
    return merged


def _merge_ball_fragments(
    fragments: dict[int, _Fragment],
    union_find: _OverlapAwareUnionFind,
) -> int:
    ball_ids = sorted(fragment.track_id for fragment in fragments.values() if fragment.cls_id == 3)
    if not ball_ids:
        return 0
    anchor = ball_ids[0]
    return sum(int(union_find.union(anchor, track_id)) for track_id in ball_ids[1:])


def _weighted_choice(members: list[TrackInfo], attribute: str) -> int | str | None:
    votes: Counter = Counter()
    for member in members:
        value = getattr(member, attribute)
        if value is not None:
            votes[value] += max(member.n_observations, 1)
    return votes.most_common(1)[0][0] if votes else None


def _pool_track_info(
    new_track_id: int,
    members: list[TrackInfo],
) -> TrackInfo:
    observation_count = sum(max(member.n_observations, 0) for member in members)
    cls_id = int(_weighted_choice(members, "cls_id") or 0)
    team_id = _weighted_choice(members, "team_id")
    team_label = _weighted_choice(members, "team_label")

    jersey_tens = np.zeros(10, dtype=np.float32)
    jersey_units = np.zeros(10, dtype=np.float32)
    jersey_vote_count = 0
    for member in members:
        if member.jersey_logits_tens is not None:
            jersey_tens += np.asarray(member.jersey_logits_tens, dtype=np.float32)
        if member.jersey_logits_units is not None:
            jersey_units += np.asarray(member.jersey_logits_units, dtype=np.float32)
        jersey_vote_count += member.jersey_vote_count

    provisional = max(
        members,
        key=lambda member: member.jersey_confidence or 0.0,
    )
    embeddings = [
        (member.embedding_mean, max(member.n_observations, 1))
        for member in members
        if member.embedding_mean is not None
    ]
    embedding = None
    if embeddings:
        total_weight = sum(weight for _, weight in embeddings)
        mean = sum(vector * weight for vector, weight in embeddings) / total_weight
        embedding = _unit(mean)

    return TrackInfo(
        track_id=new_track_id,
        cls_id=cls_id,
        cls_name=next(
            (member.cls_name for member in members if member.cls_id == cls_id),
            members[0].cls_name,
        ),
        team_id=int(team_id) if team_id is not None else None,
        team_label=str(team_label) if team_label is not None else None,
        jersey_number=provisional.jersey_number,
        jersey_confidence=provisional.jersey_confidence,
        n_frames_visible_gate=sum(member.n_frames_visible_gate for member in members),
        first_frame=min(member.first_frame for member in members),
        last_frame=max(member.last_frame for member in members),
        embedding_mean=embedding,
        n_observations=observation_count,
        jersey_logits_tens=jersey_tens.tolist() if jersey_vote_count else None,
        jersey_logits_units=jersey_units.tolist() if jersey_vote_count else None,
        jersey_vote_count=jersey_vote_count,
    )


def _apply_mapping(state: ClipState, root_mapping: dict[int, int]) -> dict[int, int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for old_track_id, root in root_mapping.items():
        groups[root].append(old_track_id)

    ordered_roots = sorted(
        groups,
        key=lambda root: (
            min(state.tracks[track_id].first_frame for track_id in groups[root]),
            root,
        ),
    )
    compact = {root: index + 1 for index, root in enumerate(ordered_roots)}
    mapping = {old_track_id: compact[root] for old_track_id, root in root_mapping.items()}

    for observation in state.observations:
        if observation.track_id in mapping:
            observation.track_id = mapping[observation.track_id]

    new_tracks: dict[int, TrackInfo] = {}
    for root, old_track_ids in groups.items():
        new_track_id = compact[root]
        members = [state.tracks[track_id] for track_id in old_track_ids]
        new_tracks[new_track_id] = _pool_track_info(new_track_id, members)
    state.tracks = new_tracks
    return mapping


def merge_tracks(
    state: ClipState,
    *,
    embedder_source: EmbeddingSource | None = None,
    config: MergeConfig | None = None,
) -> dict[int, int]:
    """Merge offline fragments in place and return ``old_id -> new_id``.

    Team assignment and pitch calibration must run before this function. If a
    person fragment has no team, conservative behavior is to leave it unmerged.
    """
    if not state.tracks:
        return {}

    config = config or MergeConfig()
    fragments = _build_fragments(state, embedder_source)
    if not fragments:
        return {}

    for track_id, fragment in fragments.items():
        state.tracks[track_id].embedding_mean = fragment.embedding
        state.tracks[track_id].n_observations = len(fragment.frames)

    fps = max(float(state.meta.fps), 1.0)
    union_find = _OverlapAwareUnionFind(fragments)
    jersey_merges = _merge_by_jersey(fragments, union_find, config)
    motion_merges = _merge_by_motion(fragments, union_find, config, fps)
    appearance_merges = _merge_by_appearance(fragments, union_find, config, fps)
    ball_merges = _merge_ball_fragments(fragments, union_find)

    root_mapping = {track_id: union_find.find(track_id) for track_id in fragments}
    mapping = _apply_mapping(state, root_mapping)
    logger.info(
        "offline merge: %d fragments -> %d identities "
        "(jersey=%d, motion=%d, appearance=%d, ball=%d)",
        len(fragments),
        len(state.tracks),
        jersey_merges,
        motion_merges,
        appearance_merges,
        ball_merges,
    )
    return mapping
