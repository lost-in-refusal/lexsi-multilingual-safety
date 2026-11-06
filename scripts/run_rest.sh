#!/bin/sh
# Remaining steps after the probe analysis (run_all.sh stopped there once).
set -e
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
PY=.venv/bin/python
step() { echo "=== $(date +%H:%M) $*"; "$@"; }
[ -f runs/eval/base/task.gen.cap512.jsonl ] || step $PY scripts/retry_truncated.py base
step $PY -m lir.evaluate --arm base --sets task
for a in base plain safegrad; do
  step $PY -m lir.judge --input runs/eval/$a/task.gen.jsonl --output runs/eval/$a/task.judge.jsonl
done
step $PY -m lir.metrics --arms base plain safegrad
echo ALL_DONE
