"""Run the real soccer-app pipeline (detector -> DINOv3 embed -> BoT-SORT ->
jersey/team) on a video and dump its tracklets, so the honest end-to-end result
(with real detection + tracking, not GT fragments) can be scored on LTPI.

Bypasses the FastAPI/DB layer: builds the processor exactly like the job worker,
runs pass1 + aggregate, and writes two parquet files — per-frame observations and
per-track aggregates."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import pandas as pd

from app.services.clip_state import ClipState
from app.services.detector import get_detector
from app.services.video_processor import VideoProcessor, read_metadata


def build_processor():
    from app.services.embedder import get_embedder
    from app.services.jersey import get_jersey_recognizer
    from app.services.team_classifier import make_team_classifier
    from app.services.tracker import make_tracker

    detector = get_detector()
    try:
        embedder = get_embedder()
    except Exception as exc:  # noqa: BLE001
        print(f"embedder disabled: {exc}")
        embedder = None
    try:
        jersey = get_jersey_recognizer()
    except Exception as exc:  # noqa: BLE001
        print(f"jersey disabled: {exc}")
        jersey = None
    return VideoProcessor(
        detector=detector,
        tracker_factory=lambda fps: make_tracker(frame_rate=int(round(fps)) or 30, with_reid=embedder is not None),
        jersey_recognizer=jersey,
        team_classifier_factory=make_team_classifier,
        embedder=embedder,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    proc = build_processor()
    meta = read_metadata(args.video)
    print(f"meta: {meta.width}x{meta.height} {meta.fps}fps {meta.frame_count} frames")
    state = ClipState(meta=meta)

    t0 = time.time()
    proc.pass1(args.video, state)
    print(f"pass1 done: {len(state.observations)} observations, {time.time()-t0:.0f}s")
    proc.aggregate(state)
    print(f"aggregate done: {len(state.tracks)} tracks")

    obs = [dict(frame_idx=o.frame_idx, track_id=o.track_id, cls_id=o.cls_id,
                x1=o.bbox_xyxy[0], y1=o.bbox_xyxy[1], x2=o.bbox_xyxy[2], y2=o.bbox_xyxy[3],
                det_conf=o.det_confidence, team_id=o.team_id)
           for o in state.observations if o.track_id is not None]
    pd.DataFrame(obs).to_csv(out / "observations.csv", index=False)

    trk = [dict(track_id=t.track_id, team_id=t.team_id, team_label=getattr(t, "team_label", None),
                jersey_number=t.jersey_number, jersey_confidence=t.jersey_confidence,
                n_observations=t.n_observations)
           for t in state.tracks.values()]
    pd.DataFrame(trk).to_csv(out / "tracks.csv", index=False)
    print(f"wrote {len(obs)} obs, {len(trk)} tracks -> {out}")


if __name__ == "__main__":
    main()
