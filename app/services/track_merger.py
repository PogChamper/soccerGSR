"""Consolidate fragmented BoT-SORT tracks into per-player identities.

Merge signals, in priority order: same confident jersey number, high DINOv3
cosine similarity, hard cap per class (longest-K survive, rest relabelled to
the best compatible keeper). Operates in-place on ClipState; runs after
jersey vote collection and before team clustering.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.clip_state import ClipState, TrackInfo

logger = logging.getLogger(__name__)


# ---- defaults --------------------------------------------------------------

DEFAULT_MAX_PLAYERS = 24          # 11+11 outfield + a couple of subs warming up
DEFAULT_MAX_GOALKEEPERS = 2
DEFAULT_MAX_REFEREES = 4
DEFAULT_MAX_BALLS = 1
DEFAULT_COSINE_THRESHOLD = 0.86   # tuned for L2-normalised DINOv3 ViT-S+
DEFAULT_MIN_TRACK_LEN = 3         # frames; shorter tracks are always candidates to drop

# Teleport guard for appearance/hard-cap merges (jersey merges bypass it):
# refuse a merge when the temporal gap is too long or the implied screen-space
# motion across it is physically impossible. Keeps look-alikes (two GKs, two
# referees) from being glued into one identity that jumps across the pitch.
MAX_MERGE_GAP_FRAMES = 120        # ~4.8 s at 25 fps
MAX_MERGE_SPEED_PXPF = 35.0       # px/frame of bbox-centre travel across the gap
MERGE_DIST_MARGIN_PX = 160.0      # slack for camera pan/zoom on top of the speed


# ---- overlap-aware union-find ---------------------------------------------


class _OverlapAwareUF:
    """Union-find that refuses any merge whose resulting cluster would have
    two members visible in the same frame — holds even transitively through
    a chain of pairwise-allowed merges (frame-sets are merged with clusters).
    """

    def __init__(self, frames_per_track: Dict[int, set]):
        self.parent = {tid: tid for tid in frames_per_track}
        # take a copy so we can mutate freely
        self.frames = {tid: set(frames_per_track[tid]) for tid in frames_per_track}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # the two cluster frame-sets must be disjoint
        return self.frames[ra].isdisjoint(self.frames[rb])

    def union(self, a, b) -> bool:
        if not self.can_union(a, b):
            return False
        ra, rb = self.find(a), self.find(b)
        # canonical id = lower (for stable, deterministic remap)
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.frames[ra].update(self.frames[rb])
        del self.frames[rb]
        return True


# ---- helpers ---------------------------------------------------------------


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))


def _track_endpoints(state: ClipState) -> Dict[int, Tuple[int, int, Tuple[float, float], Tuple[float, float]]]:
    """Per-track (first_frame, last_frame, first_centre, last_centre) in screen px."""
    first: Dict[int, Tuple[int, Tuple[float, float]]] = {}
    last: Dict[int, Tuple[int, Tuple[float, float]]] = {}
    for o in state.observations:
        tid = o.track_id
        if tid is None:
            continue
        cx = (o.bbox_xyxy[0] + o.bbox_xyxy[2]) * 0.5
        cy = (o.bbox_xyxy[1] + o.bbox_xyxy[3]) * 0.5
        if tid not in first or o.frame_idx < first[tid][0]:
            first[tid] = (o.frame_idx, (cx, cy))
        if tid not in last or o.frame_idx > last[tid][0]:
            last[tid] = (o.frame_idx, (cx, cy))
    out: Dict[int, Tuple[int, int, Tuple[float, float], Tuple[float, float]]] = {}
    for tid in first:
        out[tid] = (first[tid][0], last[tid][0], first[tid][1], last[tid][1])
    return out


def _merge_compatible(a: int, b: int, endp: dict) -> bool:
    """Reject appearance merges that would imply a teleport between fragments."""
    ea, eb = endp.get(a), endp.get(b)
    if ea is None or eb is None:
        return True
    # order by time
    if ea[0] <= eb[0]:
        earlier, later = ea, eb
    else:
        earlier, later = eb, ea
    gap = later[0] - earlier[1]            # later.first - earlier.last
    if gap <= 0:
        return True                         # overlapping in time -> UF blocks it
    if gap > MAX_MERGE_GAP_FRAMES:
        return False
    ex, ey = earlier[3]                      # earlier.last_centre
    lx, ly = later[2]                        # later.first_centre
    dist = ((lx - ex) ** 2 + (ly - ey) ** 2) ** 0.5
    return dist <= MAX_MERGE_SPEED_PXPF * gap + MERGE_DIST_MARGIN_PX


# ---- main API --------------------------------------------------------------


def merge_tracks(
    state: ClipState,
    *,
    embedder_source=None,
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    max_players: int = DEFAULT_MAX_PLAYERS,
    max_goalkeepers: int = DEFAULT_MAX_GOALKEEPERS,
    max_referees: int = DEFAULT_MAX_REFEREES,
    max_balls: int = DEFAULT_MAX_BALLS,
    min_track_len: int = DEFAULT_MIN_TRACK_LEN,
) -> Dict[int, int]:
    """Consolidate fragmented tracks in ``state`` in-place.

    ``embedder_source`` should expose a ``mean_embedding(track_id) ->
    Optional[np.ndarray]`` callable (TeamClassifier satisfies this).
    Returns the {old_track_id -> new_track_id} mapping that was applied.
    """
    if not state.tracks:
        return {}

    # 1. Collect per-track summary that the merger needs.
    embeddings: Dict[int, np.ndarray] = {}
    if embedder_source is not None:
        for tid in state.tracks:
            emb = embedder_source.mean_embedding(tid)
            if emb is not None:
                embeddings[tid] = emb
                state.tracks[tid].embedding_mean = emb  # cache for later use

    # n_observations needs to be computed once
    obs_count = Counter(o.track_id for o in state.observations if o.track_id is not None)
    for tid, ti in state.tracks.items():
        ti.n_observations = obs_count.get(tid, 0)

    # 2. Compute exact frame-set per track (correct intervals; observations
    #    can have gaps) and build the overlap-aware union-find.
    frames_per_track: Dict[int, set] = defaultdict(set)
    for o in state.observations:
        if o.track_id is not None:
            frames_per_track[o.track_id].add(o.frame_idx)
    # ensure every track has an entry (defensive)
    for tid in state.tracks:
        frames_per_track.setdefault(tid, set())

    by_cls: Dict[int, List[int]] = defaultdict(list)
    for tid, ti in state.tracks.items():
        by_cls[ti.cls_id].append(tid)

    endp = _track_endpoints(state)
    uf = _OverlapAwareUF(frames_per_track)

    for cls_id, tids in by_cls.items():
        # ---- step 2a: jersey-number merging (only for players + goalkeepers)
        if cls_id in (0, 1):
            by_jersey: Dict[int, List[int]] = defaultdict(list)
            for tid in tids:
                ti = state.tracks[tid]
                if ti.jersey_number is not None and (ti.jersey_confidence or 0) > 0.65:
                    by_jersey[ti.jersey_number].append(tid)
            for jersey, group in by_jersey.items():
                # union-find handles transitive overlap correctness
                group_sorted = sorted(group, key=lambda t: state.tracks[t].first_frame)
                for i in range(len(group_sorted)):
                    for j in range(i + 1, len(group_sorted)):
                        a, b = group_sorted[i], group_sorted[j]
                        if uf.union(a, b):
                            logger.debug(f"merge by jersey #{jersey}: {a}<-{b}")

        # ---- step 2b: embedding similarity (skip ball; ball has no useful emb)
        if cls_id != 3:
            cand = [t for t in tids if t in embeddings]
            # sort by first_frame for deterministic merging order; greedily try
            # the highest-similarity allowed merge first
            pairs = []
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    a, b = cand[i], cand[j]
                    s = _cos(embeddings[a], embeddings[b])
                    if s >= cosine_threshold and _merge_compatible(a, b, endp):
                        pairs.append((s, a, b))
            pairs.sort(reverse=True)
            for s, a, b in pairs:
                if uf.union(a, b):
                    logger.debug(
                        f"merge by embedding cls={cls_id}: {a}<-{b} cos={s:.2f}"
                    )

        # ---- step 2c: ball — collapse all into a single canonical track
        if cls_id == 3 and len(tids) > 1:
            anchor = tids[0]
            for t in tids[1:]:
                uf.union(anchor, t)

    # 3. Apply the mapping (old -> new canonical).
    mapping_initial = {tid: uf.find(tid) for tid in state.tracks}
    n_after_merge = len(set(mapping_initial.values()))
    logger.info(
        f"track_merger: {len(state.tracks)} fragments -> {n_after_merge} after "
        f"(jersey + embedding + ball-collapse)"
    )

    # 4. Hard cap per (cls, team) — operate on the merged groups.
    #    We cluster by cls_id only (team_id not yet assigned at this stage).
    group_sizes = {root: 0 for root in set(mapping_initial.values())}
    group_cls = {root: state.tracks[root].cls_id for root in group_sizes}
    for tid, root in mapping_initial.items():
        group_sizes[root] += state.tracks[tid].n_observations

    by_cls_groups: Dict[int, List[int]] = defaultdict(list)
    for root, c in group_cls.items():
        by_cls_groups[c].append(root)

    cls_caps = {0: max_players, 1: max_goalkeepers, 2: max_referees, 3: max_balls}

    # root-level endpoints (for the hard-cap teleport guard)
    root_members: Dict[int, List[int]] = defaultdict(list)
    for tid, root in mapping_initial.items():
        root_members[root].append(tid)
    root_endp: Dict[int, Tuple[int, int, Tuple[float, float], Tuple[float, float]]] = {}
    for root, members in root_members.items():
        eps = [endp[t] for t in members if t in endp]
        if not eps:
            continue
        ff = min(e[0] for e in eps)
        lf = max(e[1] for e in eps)
        fc = min(eps, key=lambda e: e[0])[2]
        lc = max(eps, key=lambda e: e[1])[3]
        root_endp[root] = (ff, lf, fc, lc)

    final_mapping = dict(mapping_initial)
    for c, roots in by_cls_groups.items():
        cap = cls_caps.get(c, 32)
        if len(roots) <= cap:
            continue
        # keep top-K by length
        roots_sorted = sorted(roots, key=lambda r: -group_sizes[r])
        keepers = roots_sorted[:cap]
        droppers = roots_sorted[cap:]
        # for each dropper, find the most similar keeper that is ALSO motion-
        # compatible (no teleport). If none is compatible, leave the dropper as
        # its own identity rather than fabricating an impossible jump.
        for d in droppers:
            best = None
            best_score = -1.0
            d_emb = embeddings.get(d)
            for k in keepers:
                if not _merge_compatible(d, k, root_endp):
                    continue
                k_emb = embeddings.get(k)
                s = _cos(d_emb, k_emb) if (d_emb is not None and k_emb is not None) else 0.0
                if s > best_score:
                    best_score = s
                    best = k
            if best is None:
                continue  # keep dropper as a separate surviving identity
            for tid in list(final_mapping):
                if final_mapping[tid] == d:
                    final_mapping[tid] = best
            logger.debug(
                f"hard-cap cls={c}: {d} -> {best} (cos={best_score:.2f})"
            )

    # 5. Drop very short fragments that survived (noise tracks of len < min)
    surviving_roots = {root for root in final_mapping.values()}
    drop_roots = set()
    for root in surviving_roots:
        members = [t for t, r in final_mapping.items() if r == root]
        total = sum(state.tracks[t].n_observations for t in members)
        if total < min_track_len:
            drop_roots.add(root)
    for tid in list(final_mapping):
        if final_mapping[tid] in drop_roots:
            final_mapping[tid] = -1  # marker for "drop this observation"

    # 6. Write back to state.observations and state.tracks.
    keep_mapping = {old: new for old, new in final_mapping.items() if new != -1}
    for obs in state.observations:
        if obs.track_id is None:
            continue
        new_tid = final_mapping.get(obs.track_id, obs.track_id)
        if new_tid == -1:
            obs.track_id = None
            continue
        obs.track_id = new_tid

    # Rebuild state.tracks: collapse merged fragments.
    new_tracks: Dict[int, TrackInfo] = {}
    by_root: Dict[int, List[TrackInfo]] = defaultdict(list)
    for old_tid, ti in state.tracks.items():
        new_tid = final_mapping[old_tid]
        if new_tid == -1:
            continue
        by_root[new_tid].append(ti)

    for new_tid, members in by_root.items():
        # canonical = the one whose track_id == new_tid
        canon = next((m for m in members if m.track_id == new_tid), members[0])
        cls_id = Counter([m.cls_id for m in members]).most_common(1)[0][0]
        first = min(m.first_frame for m in members)
        last = max(m.last_frame for m in members)
        n_obs = sum(m.n_observations for m in members)
        n_vis = sum(m.n_frames_visible_gate for m in members)

        # Pool jersey votes across all fragments of this identity so the
        # authoritative number can be (re)committed from the combined
        # evidence. A provisional number is also carried for any interim use.
        pooled_votes: Dict[int, float] = defaultdict(float)
        pooled_counts: Dict[int, int] = defaultdict(int)
        for m in members:
            for num, w in (m.jersey_votes or {}).items():
                pooled_votes[num] += w
            for num, c in (m.jersey_vote_counts or {}).items():
                pooled_counts[num] += c

        jersey = None
        jersey_conf = None
        for m in sorted(
            members,
            key=lambda x: (x.jersey_confidence or 0.0),
            reverse=True,
        ):
            if m.jersey_number is not None:
                jersey = m.jersey_number
                jersey_conf = m.jersey_confidence
                break

        # mean of embedding means (re-normalised)
        emb = None
        emb_list = [m.embedding_mean for m in members if m.embedding_mean is not None]
        if emb_list:
            agg = np.mean(np.stack(emb_list), axis=0)
            n = float(np.linalg.norm(agg))
            if n > 1e-12:
                emb = (agg / n).astype(np.float32)

        new_tracks[new_tid] = TrackInfo(
            track_id=new_tid,
            cls_id=cls_id,
            cls_name=canon.cls_name,
            first_frame=first,
            last_frame=last,
            n_frames_visible_gate=n_vis,
            jersey_number=jersey,
            jersey_confidence=jersey_conf,
            embedding_mean=emb,
            n_observations=n_obs,
            jersey_votes=dict(pooled_votes),
            jersey_vote_counts=dict(pooled_counts),
        )

    # 7. Compact renumbering: relabel surviving identities to a dense 1..N
    #    range (ordered by first appearance) so downstream IDs never look like
    #    "hundreds of tracks" even when the raw tracker churned through high
    #    ids. Compose this with keep_mapping so callers remap in one shot.
    ordered = sorted(new_tracks.values(), key=lambda t: (t.first_frame, t.track_id))
    renum: Dict[int, int] = {t.track_id: i + 1 for i, t in enumerate(ordered)}

    renamed: Dict[int, TrackInfo] = {}
    for old_root, ti in new_tracks.items():
        ti.track_id = renum[old_root]
        renamed[ti.track_id] = ti
    for obs in state.observations:
        if obs.track_id is not None and obs.track_id in renum:
            obs.track_id = renum[obs.track_id]
    keep_mapping = {old: renum[new] for old, new in keep_mapping.items() if new in renum}

    state.tracks = renamed
    return keep_mapping
