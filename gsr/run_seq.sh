#!/usr/bin/env bash
# Run all three compute stages for one GSR sequence across the two conda envs.
#   run_seq.sh <split> <seq> [--relink]
set -euo pipefail
SPLIT="$1"; SEQ="$2"; shift 2 || true
EXTRA="${*:-}"
SA=/home/dxdxxd/projects/soccer-app
EDA=/home/dxdxxd/miniconda3/envs/soccer_eda/bin/python
DFR=/home/dxdxxd/miniconda3/envs/dfine-reid/bin/python

cd "$SA"
MODEL_AUTO_DOWNLOAD=false "$EDA" gsr/s1_detect_jersey.py "$SPLIT" "$SEQ"
"$DFR" gsr/s2_embed_calib.py "$SPLIT" "$SEQ"
"$EDA" gsr/s3_assemble.py "$SPLIT" "$SEQ" $EXTRA
