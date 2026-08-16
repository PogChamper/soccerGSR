#!/usr/bin/env bash
# Re-run Stage 3 (assemble) under a config tag on already-extracted features and
# score GS-HOTA. Cheap lever iteration — no detector/reid/calib recompute.
#   iter.sh <split> <seqs|all> <tag> [s3 flags...]
# Jersey knobs via env: GSR_VIS_TH GSR_OCR_CONF_TH GSR_MIN_VOTES
set -euo pipefail
SPLIT="$1"; SEQS="$2"; TAG="$3"; shift 3 || true
SA=/home/dxdxxd/projects/soccer-app
EDA=/home/dxdxxd/miniconda3/envs/soccer_eda/bin/python
SN=/home/dxdxxd/projects/soccer/sn-gamestate

cd "$SA"
GSR_TAG="$TAG" "$EDA" gsr/s3b_assemble.py "$SPLIT" "$SEQS" "$@"
cd "$SN"
PYTHONPATH=/home/dxdxxd/projects/AuxFlow/AuxFlow/.repro_deps \
  .venv/bin/python "$SA/gsr/eval_gshota.py" "$SPLIT" "$SEQS" "$TAG"
