#!/bin/zsh
# Follow the job queue live: shows each job as it starts and tails its log. Ctrl-C to stop watching;
# that does not stop the queue.
dir=${1:-runs/queue}
pid=
trap 'kill $pid 2>/dev/null; exit 0' INT TERM
while true; do
  n=$(cat $dir/next 2>/dev/null || echo 0)
  if [[ ! -f $dir/logs/$n.log ]]; then
    sleep 5
    continue
  fi
  echo
  echo "=== job $n of $(grep -c '' $dir/jobs.txt), started $(grep "start $n:" $dir/status.log | cut -c12-19)"
  echo "    $(sed -n "${n}p" $dir/jobs.txt)"
  tail -n 2 -f $dir/logs/$n.log &
  pid=$!
  while [[ "$(cat $dir/next)" == "$n" ]]; do sleep 5; done
  sleep 3
  kill $pid 2>/dev/null
  echo
  grep "end $n " $dir/status.log
done
