"""Torso-colour team classifier (pooled KMeans + referee outlier).

Pass1 accumulates per track: DINOv3 embeddings (consumed by the track-merger
for ReID) and grass-masked HS torso histograms. Aggregate clusters the
pooled histograms into two teams, assigns GKs to the nearest centroid and
promotes far-from-both-centroids tracks to referee.

Torso colour, not DINOv3, decides teams: full-crop embeddings mix in pose,
grass and lighting, and collapse on dark look-alike kits (observed 1/20
split on a night clip), while the masked HS histogram isolates the kit
colour and clusters cleanly.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.clip_state import ClipState

logger = logging.getLogger(__name__)


PER_TRACK_SAMPLE_CAP = 30
PER_TRACK_COLOR_CAP = 40

# Torso colour histogram config
HUE_BINS = 12
SAT_BINS = 3
COLOR_FEAT_DIM = HUE_BINS * SAT_BINS

# Referee promotion: how many times the median nearest-centroid distance a
# track must exceed to be considered an out-of-team kit (referee).
REFEREE_DIST_FACTOR = 2.2
# Minimum players that must remain in each team before we allow any referee
# promotion (guards against stripping a real team on sparse clips).
MIN_TEAM_SIZE_FOR_PROMOTION = 3


def torso_color_hist(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """Grass-masked HS colour histogram of a player's torso region.

    Returns an L1-normalised ``(HUE_BINS*SAT_BINS,)`` vector, or None if the
    crop is too small / has too few valid (non-grass, non-dark) pixels.
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w_img, int(x2)); y2 = min(h_img, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    roi = crop[int(0.15 * ch):int(0.55 * ch), int(0.20 * cw):int(0.80 * cw)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (H >= 35) & (H <= 85) & (S > 40)
    valid = (~grass) & (V > 30) & (V < 250)
    if int(valid.sum()) < 20:
        return None
    hist, _, _ = np.histogram2d(
        H[valid].astype(np.float32),
        S[valid].astype(np.float32),
        bins=[HUE_BINS, SAT_BINS],
        range=[[0, 180], [0, 256]],
    )
    hist = hist.flatten().astype(np.float32)
    s = hist.sum()
    if s < 1e-6:
        return None
    return hist / s


class TeamClassifier:
    """Per-clip instance: collect embeddings in pass1, cluster in aggregate."""

    def __init__(self) -> None:
        self._samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._color_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        # per-track class VOTES (confidence-agnostic count); majority is used
        # downstream so a referee/keeper whose *last* frame was mis-read as a
        # player is still categorised correctly.
        self._cls_votes: Dict[int, Counter] = defaultdict(Counter)

    def _majority_cls(self, track_id: int) -> int:
        votes = self._cls_votes.get(track_id)
        if not votes:
            return 0
        return votes.most_common(1)[0][0]

    @property
    def _cls(self) -> Dict[int, int]:
        """Back-compat view: {track_id -> majority class}."""
        return {tid: self._majority_cls(tid) for tid in self._cls_votes}

    # ---------------------------------------------------------------- access

    def mean_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Return the per-track L2-normalised mean embedding (or None)."""
        samples = self._samples.get(track_id)
        if not samples:
            return None
        mean = np.mean(np.stack(samples), axis=0)
        n = float(np.linalg.norm(mean))
        if n < 1e-12:
            return None
        return (mean / n).astype(np.float32)

    def mean_color(self, track_id: int) -> Optional[np.ndarray]:
        """Return the per-track L1-normalised mean torso colour histogram."""
        samples = self._color_samples.get(track_id)
        if not samples:
            return None
        mean = np.mean(np.stack(samples), axis=0)
        s = float(mean.sum())
        if s < 1e-9:
            return None
        return (mean / s).astype(np.float32)

    def has_track(self, track_id: int) -> bool:
        return track_id in self._cls_votes

    def remap_tracks(self, mapping: Dict[int, int]) -> None:
        """Apply a {old_track_id -> new_track_id} mapping in-place.

        Used by ``track_merger`` to keep team-classifier internal state in
        sync after consolidating fragmented tracks.
        """
        new_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        new_color: Dict[int, List[np.ndarray]] = defaultdict(list)
        new_votes: Dict[int, Counter] = defaultdict(Counter)
        for old_tid, samples in self._samples.items():
            new_tid = mapping.get(old_tid, old_tid)
            new_samples[new_tid].extend(samples)
        for old_tid, samples in self._color_samples.items():
            new_tid = mapping.get(old_tid, old_tid)
            new_color[new_tid].extend(samples)
        for old_tid, votes in self._cls_votes.items():
            new_tid = mapping.get(old_tid, old_tid)
            new_votes[new_tid].update(votes)
        self._samples = new_samples
        self._color_samples = new_color
        self._cls_votes = new_votes

    # ----------------------------------------------------------------- pass1

    def observe(
        self,
        track_id: int,
        embedding: Optional[np.ndarray],
        cls_id: int,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Accumulate per-track appearance features.

        ``embedding`` (DINOv3, for ReID/merge) may be None. If ``frame`` and
        ``bbox`` are given, a torso colour histogram is computed and stored
        for team clustering.
        """
        self._cls_votes[track_id][cls_id] += 1

        if embedding is not None and np.any(embedding):
            if len(self._samples[track_id]) < PER_TRACK_SAMPLE_CAP:
                self._samples[track_id].append(np.asarray(embedding, dtype=np.float32))

        if frame is not None and bbox is not None:
            if len(self._color_samples[track_id]) < PER_TRACK_COLOR_CAP:
                hist = torso_color_hist(frame, bbox)
                if hist is not None:
                    self._color_samples[track_id].append(hist)

    # ------------------------------------------------------------ aggregate

    def fit_and_assign(self, state: ClipState) -> None:
        """KMeans(k=2) on ALL pooled detection-level histograms (stable
        centroids from hundreds of samples; clustering ~20 track means is
        unstable on hard footage), then per-track majority vote. Outliers
        from both centroids become referees, GKs go to the nearest team."""
        from sklearn.cluster import KMeans

        cls_map = self._cls  # snapshot {tid -> majority class}
        outfield_tracks: List[int] = []
        gk_tracks: List[int] = []
        for tid in cls_map:
            if not self._color_samples.get(tid):
                continue
            cls = cls_map.get(tid, 0)
            if cls == 0:
                outfield_tracks.append(tid)
            elif cls == 1:
                gk_tracks.append(tid)

        team_assignment: Dict[int, int] = {}
        team_label: Dict[int, str] = {}
        promoted_to_referee: List[int] = []
        centroids: Optional[np.ndarray] = None
        remap: Dict[int, int] = {0: 0, 1: 1}

        # --- pool outfield detection-level colour samples ---
        if len(outfield_tracks) >= 2:
            pooled: List[np.ndarray] = []
            owner: List[int] = []
            for tid in outfield_tracks:
                for s in self._color_samples[tid]:
                    pooled.append(s)
                    owner.append(tid)
            P = np.stack(pooled)
            owner_arr = np.asarray(owner)

            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(P)
            sample_labels = km.labels_
            centroids = km.cluster_centers_

            # deterministic team ordering by cluster mean hue
            hue_axis = np.arange(HUE_BINS).repeat(SAT_BINS).astype(np.float32)
            cluster_hue = [
                float((centroids[c] / max(centroids[c].sum(), 1e-9) * hue_axis).sum())
                for c in range(2)
            ]
            order = np.argsort(cluster_hue)
            remap = {int(old): int(new) for new, old in enumerate(order)}

            # per-sample distance to nearest centroid (for referee detection)
            d_all = np.linalg.norm(P[:, None, :] - centroids[None, :, :], axis=2)
            d_nearest = d_all.min(axis=1)
            med = float(np.median(d_nearest)) or 1e-6

            # per-track: majority vote + mean outlier distance
            track_team: Dict[int, int] = {}
            track_outlier: Dict[int, float] = {}
            for tid in outfield_tracks:
                m = owner_arr == tid
                votes = sample_labels[m]
                raw = int(round(votes.mean()))  # 0/1 majority
                track_team[tid] = remap[raw]
                track_outlier[tid] = float(d_nearest[m].mean())

            size = [sum(1 for v in track_team.values() if v == 0),
                    sum(1 for v in track_team.values() if v == 1)]

            for tid in outfield_tracks:
                t = track_team[tid]
                is_ref = (
                    track_outlier[tid] > REFEREE_DIST_FACTOR * med
                    and min(size) > MIN_TEAM_SIZE_FOR_PROMOTION
                )
                if is_ref:
                    promoted_to_referee.append(tid)
                    team_label[tid] = "referee"
                    size[t] -= 1
                    continue
                team_assignment[tid] = t
                team_label[tid] = f"team_{'ab'[t]}"
        elif len(outfield_tracks) == 1:
            tid = outfield_tracks[0]
            team_assignment[tid] = 0
            team_label[tid] = "team_a"

        # --- goalkeepers: majority vote against the two team centroids ---
        if gk_tracks and centroids is not None:
            for tid in gk_tracks:
                samples = np.stack(self._color_samples[tid])
                d = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
                raw = int(round(d.argmin(axis=1).mean()))
                team = remap[raw]
                team_assignment[tid] = team
                team_label[tid] = f"goalkeeper_{'ab'[team]}"
        else:
            for tid in gk_tracks:
                team_assignment[tid] = 0
                team_label[tid] = "goalkeeper"

        # --- write to state.tracks ---
        for tid, ti in state.tracks.items():
            if tid in team_label:
                ti.team_label = team_label[tid]
                ti.team_id = team_assignment.get(tid)  # None for promoted referees
            elif ti.cls_id == 2:
                ti.team_label = "referee"
            elif ti.cls_id == 3:
                ti.team_label = "ball"

        # --- propagate team_id to observations ---
        for obs in state.observations:
            tid = obs.track_id
            if tid is None:
                continue
            if tid in team_assignment:
                obs.team_id = team_assignment[tid]

        n_a = sum(1 for v in team_assignment.values() if v == 0)
        n_b = sum(1 for v in team_assignment.values() if v == 1)
        logger.info(
            f"team classifier: pooled torso-colour KMeans + track vote — "
            f"outfield={len(outfield_tracks)} (team_a={n_a}, team_b={n_b}, "
            f"ref-promoted={len(promoted_to_referee)}), GK={len(gk_tracks)}"
        )


def make_team_classifier() -> TeamClassifier:
    return TeamClassifier()
