#!/bin/zsh
# Tiny Aya Earth runs, one step per call so each can be a line in the job queue:
#   scripts/earth.sh base | refusals | plain | safegrad | eval-plain | eval-safegrad | judge <arm> | metrics
# Every step sets the model itself, and SafeGrad always trains on Earth's own refusal set.
set -e
export LIR_MODEL=earth HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=${PY:-.venv/bin/python}
align=data/lir/alignment_sft_earth.jsonl
case $1 in
  base)          $PY -m lir.evaluate --arm earth-base --sets alignment attack xstest task ;;
  refusals)      $PY -m lir.stage2 refusals --arm earth-base --out ${align:t} ;;
  plain)         $PY -m lir.train --arm plain --output runs/earth-plain ;;
  safegrad)      [[ -s $align ]] || { echo "$align is missing: run the refusals step first"; exit 1; }
                 $PY -m lir.train --arm safegrad --output runs/earth-safegrad --align-file $align ;;
  eval-plain)    $PY -m lir.evaluate --arm earth-plain --adapter runs/earth-plain/adapter ;;
  eval-safegrad) $PY -m lir.evaluate --arm earth-safegrad --adapter runs/earth-safegrad/adapter ;;
  judge)         $PY -m lir.judge --input runs/eval/earth-$2/task.gen.jsonl --output runs/eval/earth-$2/task.judge.jsonl ;;
  metrics)       $PY -m lir.metrics --arms earth-base earth-plain earth-safegrad --ref earth-base --out results/lir/stage2/earth ;;
  *)             echo "unknown step: $1"; exit 2 ;;
esac
