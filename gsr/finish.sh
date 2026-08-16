#!/usr/bin/env bash
# After s1+s2 features exist, run association (s3a) + assemble (s3b, best config)
# + GS-HOTA eval for a split. Usage: finish.sh <split> [seqs|all]
set -euo pipefail
SPLIT="$1"; SEQS="${2:-all}"
SA=/home/dxdxxd/projects/soccer-app
EDA=/home/dxdxxd/miniconda3/envs/soccer_eda/bin/python
SN=/home/dxdxxd/projects/soccer/sn-gamestate

cd "$SA"
echo "=== s3a track ($SPLIT) ==="
"$EDA" gsr/s3a_track.py "$SPLIT" "$SEQS"
echo "=== s3b assemble ($SPLIT, best config) ==="
"$EDA" gsr/s3b_assemble.py "$SPLIT" "$SEQS"
echo "=== GS-HOTA ($SPLIT) ==="
cd "$SN"
PYTHONPATH=/home/dxdxxd/projects/AuxFlow/AuxFlow/.repro_deps \
  .venv/bin/python "$SA/gsr/eval_gshota.py" "$SPLIT" all ours
