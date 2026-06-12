"""Run aggregate + pass2 render from a cached pass1 pickle (no GPU needed).

Lets us iterate on calibration / ball filtering / minimap rendering and see
the actual output video in ~1-2 min instead of re-running the 8 min pass1.

Usage:
    python scripts/render_from_cache.py /tmp/chelleed_pass1.pkl \
        chelleed_49m45-50m45.mp4 /tmp/chelleed_render.mp4
"""
from __future__ import annotations

import copy
import logging
import pickle
import sys
import time
from collections import Counter


def main():
    pkl, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    state = copy.deepcopy(d["state"])
    team = copy.deepcopy(d["team"])

    # Cache compat: older caches stored last-frame class in `_cls`. Rebuild the
    # per-track class VOTES from raw observations so the team classifier uses a
    # true majority class (matches the live pipeline).
    from collections import Counter as _C, defaultdict as _dd
    team._cls_votes = _dd(_C)
    for _o in state.observations:
        if _o.track_id is not None:
            team._cls_votes[_o.track_id][_o.cls_id] += 1
    team.__dict__.pop("_cls", None)

    from app.services.clip_state import TrackInfo
    from app.services.jersey import JerseyRecognizer
    from app.services.track_merger import merge_tracks
    from app.services.calibration import make_calibrator
    from app.services.track_smoother import (
        smooth_pitch_xy, filter_ball_outliers, interpolate_track_gaps,
        smooth_detection_class,
    )
    from app.services.minimap import get_minimap_renderer, reset_minimap_renderer
    from app.services.video_processor import VideoProcessor, _class_name, majority_class

    # build TrackInfo
    by_track = state.observations_by_track()
    for tid, obs_list in by_track.items():
        obs_list.sort(key=lambda o: o.frame_idx)
        cls_majority = majority_class(obs_list)
        state.tracks[tid] = TrackInfo(
            track_id=tid, cls_id=cls_majority, cls_name=_class_name(cls_majority),
            first_frame=obs_list[0].frame_idx, last_frame=obs_list[-1].frame_idx,
            n_frames_visible_gate=sum(
                1 for o in obs_list
                if o.visibility_p is not None and o.visibility_p > 0.5),
        )

    JerseyRecognizer.collect_votes_into_tracks(None, state)
    mapping = merge_tracks(state, embedder_source=team)
    if mapping:
        team.remap_tracks(mapping)
    JerseyRecognizer.commit_numbers(None, state)

    t = time.time()
    cal = make_calibrator(state.meta.width, state.meta.height)
    cal.calibrate(state)
    print(f"calibrate: {time.time()-t:.1f}s")
    team.fit_and_assign(state)
    filter_ball_outliers(state)
    smooth_pitch_xy(state)
    smooth_detection_class(state)
    interpolate_track_gaps(state)

    reset_minimap_renderer()
    proc = VideoProcessor(detector=object(), minimap_renderer=get_minimap_renderer())
    t = time.time()
    proc.pass2(src, out, state, draw_legend_flag=True)
    print(f"pass2: {time.time()-t:.1f}s -> {out}")


if __name__ == "__main__":
    main()
