#!/usr/bin/env bash
# Run all three compute stages for a set of GSR sequences, loading each stage's
# models once. Pass a comma-separated seq list or the literal "all".
#   run_batch.sh <split> <seqs|all> [--relink]
set -euo pipefail
SPLIT="$1"; SEQS="$2"; shift 2 || true
EXTRA="${*:-}"
SA=/home/dxdxxd/projects/soccer-app
EDA=/home/dxdxxd/miniconda3/envs/soccer_eda/bin/python
DFR=/home/dxdxxd/miniconda3/envs/dfine-reid/bin/python

cd "$SA"
echo "=== stage 1 (detect+jersey) ==="
MODEL_AUTO_DOWNLOAD=false "$EDA" gsr/s1_detect_jersey.py "$SPLIT" "$SEQS"
echo "=== stage 2 (embed+calib) ==="
"$DFR" gsr/s2_embed_calib.py "$SPLIT" "$SEQS"
echo "=== stage 3 (assemble) ==="
"$EDA" gsr/s3_assemble.py "$SPLIT" "$SEQS" $EXTRA
