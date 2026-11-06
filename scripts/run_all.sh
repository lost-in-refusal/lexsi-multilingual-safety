#!/bin/sh
# Remaining pipeline after data freeze, one GPU job at a time. Every generation step resumes.
set -e
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
step() { echo "=== $(date +%H:%M) $*"; "$@"; }

step $PY -m lir.evaluate --arm base --sets xstest
[ -f runs/plain/run.json ] || { rm -rf runs/plain; step $PY -m lir.train --arm plain --output runs/plain; }
step $PY -m lir.evaluate --arm plain --adapter runs/plain/adapter --sets attack
step $PY -m lir.metrics --arms base plain
echo GATE_READY
[ -f runs/safegrad/run.json ] || { rm -rf runs/safegrad; step $PY -m lir.train --arm safegrad --output runs/safegrad; }
step $PY -m lir.evaluate --arm safegrad --adapter runs/safegrad/adapter --sets attack
step $PY -m lir.metrics --arms base plain safegrad
step $PY -m lir.evaluate --arm plain --adapter runs/plain/adapter --sets xstest task
step $PY -m lir.evaluate --arm safegrad --adapter runs/safegrad/adapter --sets xstest task
step $PY -m lir.probe extract --arm base
step $PY -m lir.probe extract --arm plain --adapter runs/plain/adapter
step $PY -m lir.probe extract --arm safegrad --adapter runs/safegrad/adapter
step $PY -m lir.probe analyze --arms base plain safegrad
[ -f runs/eval/base/task.gen.cap512.jsonl ] || step $PY scripts/retry_truncated.py base
step $PY -m lir.evaluate --arm base --sets task
for a in base plain safegrad; do
  step $PY -m lir.judge --input runs/eval/$a/task.gen.jsonl --output runs/eval/$a/task.judge.jsonl
done
step $PY -m lir.metrics --arms base plain safegrad
echo ALL_DONE
