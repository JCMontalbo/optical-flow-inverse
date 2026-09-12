#!/bin/bash
# Full 60-cell heart grid, three parallel workers, criteria-deciding cells first. Log: experiments/heart/grid.log
cd "$(dirname "$0")/../.."
until [ -f experiments/heart/flows/stats.json ]; do sleep 30; done
echo "=== flow stats ==="; cat experiments/heart/flows/stats.json
echo "=== grid start $(date) ==="
for i in 0 1 2; do
  PYTHONIOENCODING=utf-8 python -u experiments/heart/run_grid.py --shard $i --nshards 3 2>&1 | grep --line-buffered -v Deprecat | sed -u "s/^/[shard $i] /" &
done
wait
echo "=== grid done $(date) ==="
