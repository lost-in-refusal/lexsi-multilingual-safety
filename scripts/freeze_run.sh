#!/bin/zsh
# One plain LoRA run with its attack-set evaluation, launched only before a deadline given in epoch seconds.
# After the deadline nothing is trained: the line says so and exits 0, so the queue goes on.
#   scripts/freeze_run.sh <epoch seconds> <arm> <seed> <training file>
limit=$1 arm=$2 seed=$3 file=$4
py=${PY:-.venv/bin/python}
if (( $(date +%s) >= limit )); then
  echo "not launched: $arm. It is $(date '+%F %T') and the limit was $(date -r $limit '+%F %T')."
  exit 0
fi
scripts/wait_clean.sh 30
$py -m lir.train --arm plain --seed $seed --output runs/$arm --train-file $file || exit 1
$py -m lir.evaluate --arm $arm --adapter runs/$arm/adapter --sets attack
