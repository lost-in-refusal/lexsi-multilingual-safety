#!/bin/zsh
# One plain LoRA run at seed 42 with one training setting changed (section 14h), with its evaluation on the attack set,
# XSTest and the task set, launched only before a deadline given in epoch seconds.
# After the deadline nothing is trained: the line says so and exits 0, so the queue goes on.
#   scripts/sensitivity_run.sh <epoch seconds> <arm> <flags for lir.train, for example --lr 1e-4>
limit=$1 arm=$2
shift 2
py=${PY:-.venv/bin/python}
if (( $(date +%s) >= limit )); then
  echo "not launched: $arm. It is $(date '+%F %T') and the limit was $(date -r $limit '+%F %T')."
  exit 0
fi
scripts/wait_clean.sh 30
$py -m lir.train --arm plain --seed 42 --output runs/$arm "$@" || exit 1
$py -m lir.evaluate --arm $arm --adapter runs/$arm/adapter --sets attack xstest task
