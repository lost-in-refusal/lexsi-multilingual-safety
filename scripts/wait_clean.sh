#!/bin/zsh
# Queue step before a training run: wait until no tracked file has uncommitted changes, so the run records a
# clean commit. Gives up after the given number of minutes (default 30) and lets the queue go on.
#   scripts/wait_clean.sh 30
limit=${1:-30}
for i in $(seq $limit); do
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] && { echo "clean after $((i - 1)) min"; exit 0; }
  sleep 60
done
echo "still uncommitted changes after $limit min:"; git status --porcelain --untracked-files=no
exit 0
