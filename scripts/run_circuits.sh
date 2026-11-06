#!/bin/sh
# CircuitKIT refusal circuits for the three arms, in the order we ran them (one GPU job at a time).
# Discovery is ~15 min per language on an M5 Pro (64 pairs, EAP-IG, 5 steps); faithfulness ~5 min per circuit size.
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CIRCUITKIT_TL_MEMORY_PATCH=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
CK=.venv-circuit/bin/python
DEV=${DEV:-mps}
step() { echo "=== $(date +%H:%M) $*"; "$@" || echo "!!! failed: $*"; }
step $CK -m lir.circuits discover --tag base --device $DEV
step $CK -m lir.circuits discover --tag plain --adapter runs/plain/adapter --device $DEV
step $CK -m lir.circuits discover --tag safegrad --adapter runs/safegrad/adapter --device $DEV
for k in 50 200 300; do step $CK -m lir.circuits compare --tags base plain safegrad --k $k; done
step $CK -m lir.circuits margin --device $DEV
step $CK -m lir.circuits faithfulness --tag base --langs en --ks 612 400 300 200 100 50 --device $DEV
echo CIRCUITS_DONE
