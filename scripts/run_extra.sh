#!/bin/sh
# After run_rest.sh: Swahili checks, steering, replication on Hannah's data, CircuitKIT. Steps are independent; a failure does not stop the rest.
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CIRCUITKIT_TL_MEMORY_PATCH=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
PY=.venv/bin/python
CK=.venv-circuit/bin/python
step() { echo "=== $(date +%H:%M) $*"; "$@" || echo "!!! failed: $*"; }
until grep -q ALL_DONE logs/run_rest.log 2>/dev/null; do
  pgrep -f "scripts/run_rest.sh" >/dev/null || { echo "run_rest.sh stopped without ALL_DONE"; exit 1; }
  sleep 60
done
step $PY -m lir.swcheck backtranslate --arms base plain safegrad
step $PY -m lir.swcheck adherence --arms base plain safegrad
step $PY -m lir.labelpack backtranslate
step $PY -m lir.steer --arm plain --adapter runs/plain/adapter --direction en --coef 1.0
step $PY -m lir.steer --arm plain --adapter runs/plain/adapter --direction native --coef 1.0
[ -f runs/plain-hannah/run.json ] || { rm -rf runs/plain-hannah; step $PY -m lir.train --arm plain --output runs/plain-hannah --train-file data/lir/replication/hannah_train.jsonl; }
step $PY -m lir.evaluate --arm plain-hannah --adapter runs/plain-hannah/adapter --sets attack xstest
step $PY -m lir.metrics --arms base plain plain-hannah safegrad --out results/lir/replication
step $CK -m lir.circuits discover --tag probe --langs en --n-examples 4 --device mps --out runs/circuits-probe
if grep -q '"seconds"' runs/circuits-probe/probe_en.json 2>/dev/null; then
  step $CK -m lir.circuits discover --tag base --langs en ko zh sw --n-examples 64 --device mps
  step $CK -m lir.circuits discover --tag plain --adapter runs/plain/adapter --langs en ko zh sw --n-examples 64 --device mps
  step $CK -m lir.circuits compare --tags base plain
fi
echo EXTRA_DONE
