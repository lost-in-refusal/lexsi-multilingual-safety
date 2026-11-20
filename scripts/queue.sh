#!/bin/zsh
# One GPU job at a time. Runs each line of <dir>/jobs.txt in order and keeps
# waiting for new lines, so jobs can be appended while it runs.
#   nohup caffeinate -dims scripts/queue.sh runs/queue > runs/queue/queue.out 2>&1 &
# A line that says STOP ends the queue. Blank lines and # comments are skipped.
dir=${1:-runs/queue}
mkdir -p $dir/logs
touch $dir/jobs.txt
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
export PY=.venv/bin/python
n=$(cat $dir/next 2>/dev/null || echo 0)
while true; do
  total=$(grep -c '' $dir/jobs.txt)
  if (( n >= total )); then
    sleep 20
    continue
  fi
  n=$((n + 1))
  echo $n > $dir/next
  cmd=$(sed -n "${n}p" $dir/jobs.txt)
  [[ -z "$cmd" || "$cmd" == \#* ]] && continue
  [[ "$cmd" == STOP ]] && break
  echo "$(date '+%F %T') start $n: $cmd" >> $dir/status.log
  eval "$cmd" > $dir/logs/$n.log 2>&1
  rc=$?
  echo "$(date '+%F %T') end $n rc=$rc" >> $dir/status.log
done
echo "$(date '+%F %T') queue stopped" >> $dir/status.log
