#!/bin/zsh
# The plain run through AlignTune (section 14d) with its evaluation, launched only before a deadline in epoch seconds.
# Training runs in the AlignTune environment; the answers and the guard run in the main one, as for every other arm.
#   scripts/aligntune_run.sh <epoch seconds>
limit=$1
py=${PY:-.venv/bin/python}
d13=.venv-d13/bin/python
scripts/wait_clean.sh 30
$d13 -m lir.train_aligntune train --before $limit
rc=$?
if (( rc == 4 )); then   # out of memory or projected to take too long: the registered fallback, once
  $d13 -m lir.train_aligntune train --before $limit --micro-batch 2
  rc=$?
fi
(( rc == 0 )) || exit $rc
$py -m lir.evaluate --arm plain-aligntune --adapter runs/plain-aligntune/adapter --sets attack xstest task
