"""Jersey number recognition: visibility gate -> two-head OCR -> per-track
weighted voting (collect per fragment, commit per merged identity, dedup per
team)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.clip_state import ClipState
from app.utils.cuda_env import get_providers
from app.utils.models_registry import ensure_model

logger = logging.getLogger(__name__)


VIS_THRESHOLD = 0.6                  # visibility classifier output threshold
MIN_PER_FRAME_OCR_CONF = 0.7         # drop per-frame OCR results below this
MIN_VOTES_FOR_NUMBER = 4             # need at least N high-conf frames to commit
# Floor on the winning number's accumulated weight (one good frame adds
# ~0.5-0.9); rejects numbers supported by a handful of weak frames.
MIN_TOP_WEIGHT = 4.0
# Winner must beat the best alternative by this ratio — the primary
# discriminator, robust to a long tail of scattered OCR noise.
MARGIN_RATIO = 2.0
# Low floor on winner's share of total votes; only rejects flat distributions
# (MARGIN_RATIO already covers the noisy-tail case).
MIN_CONFIDENCE_FOR_NUMBER = 0.35
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class JerseyResult:
    visibility_p: float
    ocr_logits_tens: Optional[List[float]] = None
    ocr_logits_units: Optional[List[float]] = None


def _crop_bgr(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 - x1 < 6 or y2 - y1 < 12:
        return None
    return frame[y1:y2, x1:x2, :]


def _to_chw_float(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 HxWx3 -> NCHW float32 with ImageNet norm at given size."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class JerseyRecognizer:
    """Singleton-friendly: holds two ONNX sessions on GPU."""

    def __init__(self):
        import onnxruntime as ort

        gate_path = ensure_model("visibility_gate")
        ocr_path = ensure_model("jersey_ocr")

        providers = get_providers(prefer_gpu=True)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        self.gate_sess = ort.InferenceSession(str(gate_path), sess_opts, providers=providers)
        self.ocr_sess = ort.InferenceSession(str(ocr_path), sess_opts, providers=providers)
        self.gate_in_name = self.gate_sess.get_inputs()[0].name
        self.ocr_in_name = self.ocr_sess.get_inputs()[0].name
        self.ocr_out_names = [o.name for o in self.ocr_sess.get_outputs()]
        logger.info(
            f"jersey: gate={gate_path.name} ocr={ocr_path.name} "
            f"providers={self.gate_sess.get_providers()}"
        )

    # ------------------------------------------------------------------ inference

    def process_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> JerseyResult:
        crop = _crop_bgr(frame, bbox)
        if crop is None:
            return JerseyResult(visibility_p=0.0)

        gate_inp = _to_chw_float(crop, 128)[None, ...]
        gate_out = self.gate_sess.run(None, {self.gate_in_name: gate_inp})[0]
        # output is (B,) logit
        if gate_out.ndim == 0:
            gate_logit = float(gate_out)
        elif gate_out.ndim == 1:
            gate_logit = float(gate_out[0])
        else:
            gate_logit = float(gate_out.reshape(-1)[0])
        p_vis = float(_sigmoid(np.array(gate_logit)))

        if p_vis < VIS_THRESHOLD:
            return JerseyResult(visibility_p=p_vis)

        ocr_inp = _to_chw_float(crop, 224)[None, ...]
        outs = self.ocr_sess.run(self.ocr_out_names, {self.ocr_in_name: ocr_inp})
        # outs[0]: (B, 10) tens, outs[1]: (B, 10) units
        logits_tens = outs[0][0].astype(np.float32).tolist()
        logits_units = outs[1][0].astype(np.float32).tolist()
        return JerseyResult(
            visibility_p=p_vis,
            ocr_logits_tens=logits_tens,
            ocr_logits_units=logits_units,
        )

    # ------------------------------------------------------------- aggregation

    @staticmethod
    def _commit_from_votes(
        votes: Dict[int, float],
        counts: Dict[int, int],
    ) -> Tuple[Optional[int], Optional[float], str]:
        """Decide a number from pooled weighted votes.

        Returns ``(number, confidence, status)``. ``status`` is "ok" or a
        rejection reason for diagnostics.
        """
        if not votes:
            return None, None, "no_votes"
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        top_num, top_w = ranked[0]
        runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
        total_w = sum(votes.values())
        agg_conf = top_w / total_w if total_w > 0 else 0.0
        margin = top_w / runner_w if runner_w > 0 else float("inf")
        n_votes_top = counts.get(top_num, 0)

        if n_votes_top < MIN_VOTES_FOR_NUMBER:
            return None, None, "too_few"
        if top_w < MIN_TOP_WEIGHT:
            return None, None, "weak"
        if runner_w > 0 and margin < MARGIN_RATIO:
            return None, None, "low_margin"
        if agg_conf < MIN_CONFIDENCE_FOR_NUMBER:
            return None, None, "low_conf"
        return top_num, agg_conf, "ok"

    def collect_votes_into_tracks(self, state: ClipState) -> None:
        """Pass A (per raw fragment, before track-merge): tally weighted OCR
        votes per number and store them on the track. The committed number
        here is provisional (a merge hint); the authoritative one comes from
        :meth:`commit_numbers` on the pooled votes of the merged identity."""
        by_track = state.observations_by_track()
        total_obs = 0
        n_passed_vis = 0
        n_passed_conf = 0

        for tid, obs_list in by_track.items():
            if tid not in state.tracks:
                continue
            track = state.tracks[tid]
            if track.cls_id not in (0, 1):
                continue

            votes: Dict[int, float] = {}
            counts: Dict[int, int] = {}
            for o in obs_list:
                total_obs += 1
                if (
                    o.visibility_p is None
                    or o.visibility_p < VIS_THRESHOLD
                    or o.ocr_logits_tens is None
                    or o.ocr_logits_units is None
                ):
                    continue
                n_passed_vis += 1
                prob_tens = _softmax(np.asarray(o.ocr_logits_tens, dtype=np.float32))
                prob_units = _softmax(np.asarray(o.ocr_logits_units, dtype=np.float32))
                tens = int(np.argmax(prob_tens))
                units = int(np.argmax(prob_units))
                c_f = float(prob_tens[tens] * prob_units[units])
                if c_f < MIN_PER_FRAME_OCR_CONF:
                    continue
                n_passed_conf += 1
                num = units if tens == 0 else tens * 10 + units
                votes[num] = votes.get(num, 0.0) + float(o.visibility_p) * c_f
                counts[num] = counts.get(num, 0) + 1

            track.jersey_votes = votes
            track.jersey_vote_counts = counts
            num, conf, _ = JerseyRecognizer._commit_from_votes(votes, counts)
            track.jersey_number = num
            track.jersey_confidence = conf

        logger.info(
            f"jersey votes collected: obs={total_obs} vis_pass={n_passed_vis} "
            f"ocr_conf_pass={n_passed_conf} over {len(by_track)} fragments"
        )

    def commit_numbers(self, state: ClipState) -> None:
        """Pass B (per merged identity, after track-merge): recompute numbers
        from pooled votes. No dedup here — numbers are only unique within a
        team, so that waits for team assignment (:meth:`dedup_numbers`)."""
        n_ok = 0
        reasons: Dict[str, int] = {}
        for t in state.tracks.values():
            if t.cls_id not in (0, 1):
                continue
            votes = t.jersey_votes or {}
            counts = t.jersey_vote_counts or {}
            num, conf, status = JerseyRecognizer._commit_from_votes(votes, counts)
            reasons[status] = reasons.get(status, 0) + 1
            t.jersey_number = num
            t.jersey_confidence = conf
            if num is not None:
                n_ok += 1

        logger.info(
            f"jersey commit: {n_ok}/{len(state.tracks)} tracks got numbers "
            f"(reasons={reasons})"
        )

    def dedup_numbers(self, state: ClipState) -> None:
        """Pass C (after team assignment): within each (team, number) group
        keep the track with the most accumulated vote weight — a long,
        well-observed track beats a short over-confident fragment. Different
        teams may legitimately share a number."""
        def _team_key(t) -> object:
            if t.team_id is not None:
                return t.team_id
            return t.team_label or "unknown"

        by_team_number: Dict[Tuple[object, int], List[int]] = {}
        for tid, t in state.tracks.items():
            if t.jersey_number is None:
                continue
            key = (_team_key(t), t.jersey_number)
            by_team_number.setdefault(key, []).append(tid)

        def _evidence(tid: int) -> float:
            t = state.tracks[tid]
            return (t.jersey_votes or {}).get(t.jersey_number, 0.0)

        n_dedup = 0
        for (team, num), tids in by_team_number.items():
            if len(tids) <= 1:
                continue
            tids_sorted = sorted(tids, key=lambda x: -_evidence(x))
            keeper = tids_sorted[0]
            keeper_ev = _evidence(keeper)
            for loser in tids_sorted[1:]:
                lt = state.tracks[loser]
                logger.debug(
                    f"  dedup drop tid={loser} J{num} team={team} "
                    f"ev={_evidence(loser):.1f} "
                    f"(kept tid={keeper} ev={keeper_ev:.1f})"
                )
                lt.jersey_number = None
                lt.jersey_confidence = None
                n_dedup += 1

        if n_dedup:
            logger.info(f"jersey dedup: dropped {n_dedup} duplicate (team, number) claims")


_INSTANCE: Optional[JerseyRecognizer] = None


def get_jersey_recognizer() -> JerseyRecognizer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = JerseyRecognizer()
    return _INSTANCE
