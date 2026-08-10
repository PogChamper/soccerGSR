# Vendored from BoxMOT (AGPL-3.0). Modifications: dropped torch import and
# the ReidAutoBackend integration. ReID is no longer built-in; pass
# pre-computed appearance embeddings via the `embs` argument of `update()`.

import numpy as np

from app.vendor.boxmot.motion.cmc import get_cmc_method
from app.vendor.boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from app.vendor.boxmot.trackers.basetracker import DET_COLS, BaseTracker
from app.vendor.boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from app.vendor.boxmot.trackers.botsort.botsort_track import STrack
from app.vendor.boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from app.vendor.boxmot.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)


class BotSort(BaseTracker):
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str | None = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
    ):
        super().__init__()

        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        BaseTrack.clear_count()

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid

        cmc_cls = get_cmc_method(cmc_method)
        self.cmc = cmc_cls() if cmc_cls is not None else None
        self.fuse_first_associate = fuse_first_associate

    def _compensate_camera_motion(self, img, strack_pool, unconfirmed):
        if self.cmc is None:
            return
        warp = self.cmc.apply(img)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, embs)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        if self.with_reid and embs is None:
            raise RuntimeError(
                "BotSort: with_reid=True requires pre-computed embeddings; "
                "pass update(..., embs=...) or set with_reid=False."
            )
        features_high = embs_first if embs_first is not None else []

        detections = self._create_detections(dets_first, features_high)

        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        u_track_first, u_detection_first = self._first_association(
            unconfirmed,
            img,
            detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        )

        self._second_association(
            dets_second,
            activated_stracks,
            lost_stracks,
            refind_stracks,
            u_track_first,
            strack_pool,
        )

        u_detection_unc = self._handle_unconfirmed_tracks(
            u_detection_first,
            detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        )

        self._initialize_new_tracks(
            u_detection_unc,
            activated_stracks,
            [detections[i] for i in u_detection_first],
        )

        self._update_track_states(removed_stracks)

        return self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

    def _split_detections(self, dets, embs):
        if dets.size == 0:
            dets = np.empty((0, DET_COLS + 1), dtype=np.float32)
        else:
            det_inds = np.arange(len(dets), dtype=np.float32).reshape(-1, 1)
            dets = np.hstack([dets, det_inds])
        confs = dets[:, 4]
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) == 0:
            return []
        if self.with_reid:
            return [STrack(det, f) for (det, f) in zip(dets_first, features_high)]
        return [STrack(det) for det in dets_first]

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        STrack.multi_predict(strack_pool)
        self._compensate_camera_motion(img, strack_pool, unconfirmed)

        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        return u_track, u_detection

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
    ):
        detections_second = [STrack(det) for det in dets_second]

        r_tracked_stracks = [
            strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked
        ]

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

    def _handle_unconfirmed_tracks(
        self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed
    ):
        detections = [detections[i] for i in u_detection]

        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return u_detection

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)

    def _update_track_states(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind] for t in self.active_tracks if t.is_activated
        ]

        return (
            np.asarray(outputs, dtype=np.float32)
            if outputs
            else self.empty_output(dtype=np.float32)
        )
