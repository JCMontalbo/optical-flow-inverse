"""Run every (budget, arm, seed) cell not already in results.csv.

    python experiments/downstream/run_grid.py            # full grid
    python experiments/downstream/run_grid.py --budgets 1 2 --arms none standard flow --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BUDGETS  # noqa: E402
from train import ARMS, RESULTS, append_result, run  # noqa: E402


def done_cells():
    if not RESULTS.exists():
        return set()
    with RESULTS.open() as f:
        return {(int(r["budget"]), r["arm"], int(r["seed"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()
    cells = [(b, a, s) for s in args.seeds for b in args.budgets for a in args.arms]
    done = done_cells()
    todo = [c for c in cells if c not in done]
    print(f"{len(todo)} cells to run ({len(done)} already done)")
    t0 = time.time()
    for i, (b, a, s) in enumerate(todo, 1):
        row = run(b, a, s, max_steps=args.max_steps)
        append_result(row)
        print(f"[{i}/{len(todo)}] budget={b:2d} arm={a:14s} seed={s}  J={row['jmean']:.4f}  es={row['es_iou']:.3f}  best_step={row['best_step']:4d}  {row['minutes']:.1f} min  (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
