"""Run only pass1 on a clip and pickle (ClipState, TeamClassifier).

This lets us iterate on the aggregate stage (jersey aggregation, track
merging, team clustering, calibration, smoothing) in seconds instead of
re-running the ~8 min GPU pass1 every time.

Usage:
    python scripts/cache_pass1.py <input.mp4> <cache.pkl>
"""
from __future__ import annotations

import logging
import pickle
import sys
import time


def _block_torch():
    class _H:
        def find_spec(self, n, p=None, t=None):
            if n == "torch" or n.startswith("torch."):
                raise ImportError(n)
    sys.meta_path.insert(0, _H())


def main() -> None:
    inp = sys.argv[1]
    out = sys.argv[2]

    _block_torch()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from app.utils.cuda_env import bootstrap as cb
    cb()
    from app.utils.models_registry import ensure_models, REGISTRY
    ensure_models(list(REGISTRY))

    from app.services.detector import get_detector
    from app.services.jersey import get_jersey_recognizer
    from app.services.embedder import get_embedder
    from app.services.keypoints import get_keypoints_extractor

    det = get_detector()
    jer = get_jersey_recognizer()
    emb = get_embedder()
    kp = get_keypoints_extractor()

    from app.services.tracker import make_tracker
    from app.services.team_classifier import make_team_classifier
    from app.services.video_processor import VideoProcessor, read_metadata
    from app.services.clip_state import ClipState

    meta = read_metadata(inp)
    state = ClipState(meta=meta)
    p = VideoProcessor(
        detector=det,
        tracker_factory=lambda fps: make_tracker(int(round(fps)) or 25, with_reid=True),
        jersey_recognizer=jer,
        keypoints_extractor=kp,
        embedder=emb,
    )
    tracker = p.tracker_factory(meta.fps)
    team = make_team_classifier()

    t0 = time.time()
    p.pass1(inp, state, tracker=tracker, team_classifier=team)
    logging.info(f"pass1 done in {time.time() - t0:.1f}s, obs={len(state.observations)}")

    with open(out, "wb") as f:
        pickle.dump({"state": state, "team": team}, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"cached -> {out}")


if __name__ == "__main__":
    main()
