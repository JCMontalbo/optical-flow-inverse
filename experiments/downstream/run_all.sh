#!/bin/bash
# Wait for the offline generation to finish, then run the whole grid. Detached; log in experiments/downstream/grid.log
cd "$(dirname "$0")/../.."
# generation writes stats.json last; also require all 683 per-frame files
until [ -f experiments/downstream/generated/stats.json ] && [ "$(ls experiments/downstream/generated/*.npz | wc -l)" -ge 683 ]; do sleep 60; done
echo "=== generation stats ==="; cat experiments/downstream/generated/stats.json
echo "=== grid start $(date) ==="
# The 24 decisive cells only (docs/downstream_plan.md): H1/H2 need budgets 1,2 x arms B,C,E x 3 seeds;
# H3 needs budget 20 x arms B,C x 3 seeds.
for i in 0 1 2; do
  ( PYTHONIOENCODING=utf-8 python -u experiments/downstream/run_grid.py --shard $i --nshards 3 --budgets 1 2 --arms standard flow propagate 2>&1 | grep -v Deprecat
    PYTHONIOENCODING=utf-8 python -u experiments/downstream/run_grid.py --shard $i --nshards 3 --budgets 20 --arms standard flow 2>&1 | grep -v Deprecat
  ) | sed -u "s/^/[shard $i] /" &
done
wait
echo "=== grid done $(date) ==="
