"""Clip-level team assignment from SoccerNet OSNet track embeddings."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.services.clip_state import ClipState

logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingStats:
    total: np.ndarray
    count: int

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> _EmbeddingStats:
        return cls(np.asarray(vector, dtype=np.float64).copy(), 1)

    def add(self, vector: np.ndarray) -> None:
        self.total += np.asarray(vector, dtype=np.float64)
        self.count += 1

    def merge(self, other: _EmbeddingStats) -> None:
        self.total += other.total
        self.count += other.count

    def normalized_mean(self) -> np.ndarray | None:
        norm = float(np.linalg.norm(self.total))
        if norm <= 1e-12:
            return None
        return (self.total / norm).astype(np.float32)


def _left_cluster_by_frame(
    state: ClipState,
    track_cluster: dict[int, int],
) -> int | None:
    """Orient two anonymous clusters from calibrated per-frame positions."""
    by_frame: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_track: dict[int, list[float]] = defaultdict(list)
    for observation in state.observations:
        if observation.track_id not in track_cluster or observation.pitch_xy is None:
            continue
        cluster = track_cluster[observation.track_id]
        pitch_x = float(observation.pitch_xy[0])
        by_frame[observation.frame_idx][cluster].append(pitch_x)
        by_track[observation.track_id].append(pitch_x)

    vote = 0
    for clusters in by_frame.values():
        if not clusters.get(0) or not clusters.get(1):
            continue
        vote += 1 if np.mean(clusters[0]) <= np.mean(clusters[1]) else -1
    if vote:
        return 0 if vote > 0 else 1

    medians: dict[int, list[float]] = defaultdict(list)
    for track_id, positions in by_track.items():
        medians[track_cluster[track_id]].append(float(np.median(positions)))
    if not medians.get(0) or not medians.get(1):
        return None
    return 0 if min(medians[0]) <= min(medians[1]) else 1


def _goalkeeper_side(state: ClipState, track_id: int) -> int | None:
    positions = [
        float(observation.pitch_xy[0])
        for observation in state.observations
        if observation.track_id == track_id and observation.pitch_xy is not None
    ]
    if not positions:
        return None
    return int(float(np.median(positions)) >= 0.0)


class TeamClassifier:
    """Accumulate track features and cluster field players into two teams."""

    def __init__(self) -> None:
        self._embeddings: dict[int, _EmbeddingStats] = {}

    def observe(self, track_id: int, embedding: np.ndarray | None) -> None:
        if embedding is None:
            return
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all() or not np.any(vector):
            return
        stats = self._embeddings.get(track_id)
        if stats is None:
            self._embeddings[track_id] = _EmbeddingStats.from_vector(vector)
        else:
            stats.add(vector)

    def mean_embedding(self, track_id: int) -> np.ndarray | None:
        stats = self._embeddings.get(track_id)
        return stats.normalized_mean() if stats is not None else None

    def remap_tracks(self, mapping: dict[int, int]) -> None:
        remapped: dict[int, _EmbeddingStats] = {}
        for old_track_id, stats in self._embeddings.items():
            new_track_id = mapping.get(old_track_id, old_track_id)
            current = remapped.get(new_track_id)
            if current is None:
                remapped[new_track_id] = _EmbeddingStats(stats.total.copy(), stats.count)
            else:
                current.merge(stats)
        self._embeddings = remapped

    def fit_and_assign(self, state: ClipState) -> None:
        """Assign team IDs and spatial labels without changing track roles."""
        from sklearn.cluster import KMeans

        for track in state.tracks.values():
            track.team_id = None
            track.team_label = None
        for observation in state.observations:
            observation.team_id = None

        field_tracks = sorted(
            track_id
            for track_id, track in state.tracks.items()
            if track.cls_id in (0, 1) and self.mean_embedding(track_id) is not None
        )
        assignment: dict[int, int] = {}
        oriented = False

        if len(field_tracks) >= 2:
            features = np.stack([self.mean_embedding(track_id) for track_id in field_tracks])
            if np.unique(features, axis=0).shape[0] >= 2:
                labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(features)
                clusters = {
                    track_id: int(label)
                    for track_id, label in zip(field_tracks, labels, strict=True)
                }
                left_cluster = _left_cluster_by_frame(state, clusters)
                oriented = left_cluster is not None
                remap = (
                    {left_cluster: 0, 1 - left_cluster: 1}
                    if left_cluster is not None
                    else {0: 0, 1: 1}
                )
                assignment = {track_id: remap[cluster] for track_id, cluster in clusters.items()}
        elif len(field_tracks) == 1:
            assignment[field_tracks[0]] = 0

        if oriented:
            for track_id in field_tracks:
                if state.tracks[track_id].cls_id != 1:
                    continue
                side = _goalkeeper_side(state, track_id)
                if side is not None:
                    assignment[track_id] = side

        for track_id, track in state.tracks.items():
            if track.cls_id == 2:
                track.team_label = "referee"
                continue
            if track.cls_id == 3:
                track.team_label = "ball"
                continue
            if track_id not in assignment:
                continue

            team_id = assignment[track_id]
            track.team_id = team_id
            if track.cls_id == 1:
                labels = (
                    ("goalkeeper_left", "goalkeeper_right")
                    if oriented
                    else (
                        "goalkeeper_0",
                        "goalkeeper_1",
                    )
                )
            else:
                labels = ("team_left", "team_right") if oriented else ("team_0", "team_1")
            track.team_label = labels[team_id]

        for observation in state.observations:
            if observation.track_id in assignment:
                observation.team_id = assignment[observation.track_id]

        counts = [sum(team == index for team in assignment.values()) for index in (0, 1)]
        logger.info(
            "team clustering: tracks=%d team0=%d team1=%d orientation=%s",
            len(field_tracks),
            counts[0],
            counts[1],
            "pitch" if oriented else "anonymous",
        )


def make_team_classifier() -> TeamClassifier:
    return TeamClassifier()
