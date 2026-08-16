"""A/B one BoT-SORT config (shipped) with the A1 (freeze-EMA) / A2 (adaptive
appearance) levers, toggled by the BT_FREEZE_EMA / BT_ADAPTIVE_APP env vars (read
at import). Reports IDF1/HOTA/AssPr/AssRe/recall on the OSNet cache — the online
tracker metrics these levers target (purer tracklets feed the relink)."""
import os
import pickle
import sys

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")

import pandas as pd

from track_bench import GT, replay
from phase1_eval import full_metrics


def main():
    cache_path, video = sys.argv[1], sys.argv[2]
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gt = pd.read_csv(GT); gt["id"] = gt.team_global * 100 + gt.jersey_number
    lo, hi = cache[0][0] + 1, cache[-1][0] + 1
    gt = gt[(gt.image_id >= lo) & (gt.image_id <= hi)]
    obs = replay(cache, video, dict(appearance_thresh=0.40, match_thresh=0.80, proximity_thresh=0.50, track_buffer=90))
    m = full_metrics(obs, gt)
    tag = ("A2+" if os.environ.get("BT_ADAPTIVE_APP") == "1" else "") + ("A1" if os.environ.get("BT_FREEZE_EMA") == "1" else "")
    tag = tag or "baseline"
    print(f"{tag:<12} IDF1={m['IDF1']:.3f} HOTA={m['HOTA']:.3f} AssPr={m['AssPr']:.3f} AssRe={m['AssRe']:.3f} "
          f"IDsw={m['IDsw']} recall={m['recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
