"""GS-HOTA evaluation (run in sn-gamestate/.venv, cwd=sn-gamestate).

Scores the predictions written by Stage 3 against GT with the SoccerNetGS
TrackEval fork. GS-HOTA is printed by TrackEval literally as "HOTA".

    cd /home/dxdxxd/projects/soccer/sn-gamestate && \
      .venv/bin/python /home/dxdxxd/projects/soccer-app/gsr/eval_gshota.py <split> [seq,seq,...]
"""
import sys
from pathlib import Path

import numpy as np
import trackeval

OUT_ROOT = Path("/home/dxdxxd/projects/soccer-app/gsr/out")
GT_FOLDER = "/mnt/d/datasets/soccernet2025"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    tag = sys.argv[3] if len(sys.argv) > 3 else "ours"
    data_dir = OUT_ROOT / "preds" / f"SoccerNetGS-{split}" / tag / "data"
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        seqs = sys.argv[2].split(",")
    else:
        seqs = sorted(p.stem for p in data_dir.glob("*.json"))
    seq_info = {s: 750 for s in seqs}

    eval_config = {"USE_PARALLEL": False, "PRINT_RESULTS": True, "PRINT_ONLY_COMBINED": True,
                   "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                   "PLOT_CURVES": False, "TIME_PROGRESS": False, "BREAK_ON_ERROR": True}
    dcfg = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    dcfg.update({"GT_FOLDER": GT_FOLDER, "TRACKERS_FOLDER": str(OUT_ROOT / "preds"),
                 "SPLIT_TO_EVAL": split, "TRACKERS_TO_EVAL": [tag], "SEQ_INFO": seq_info,
                 "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                 "PLOT_CURVES": False})

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.SoccerNetGS(dcfg)]
    res, _ = evaluator.evaluate(dataset_list, [trackeval.metrics.HOTA(), trackeval.metrics.Identity()])
    d = res["SoccerNetGS"][tag]["COMBINED_SEQ"]["person"]["HOTA"]
    print(f"\n===== GS-HOTA over {len(seqs)} {split} seq(s) =====")
    print(f"GS-HOTA : {100 * float(np.mean(d['HOTA'])):.2f}")
    print(f"GS-DetA : {100 * float(np.mean(d['DetA'])):.2f}")
    print(f"GS-AssA : {100 * float(np.mean(d['AssA'])):.2f}")


if __name__ == "__main__":
    main()
