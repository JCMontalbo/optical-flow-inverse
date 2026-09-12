"""Run every (budget, arm, seed) cell not already in results.csv.

    python experiments/downstream/run_grid.py            # full grid
    python experiments/downstream/run_grid.py --shard 0 --nshards 3   # one of three parallel workers

Cells are ordered so the ones that decide the pre-registered criteria come first:
seed 0 at budgets 1 and 2 (all arms), then seed 0 at 5 and 20, then seeds 1 and 2.
Each shard appends to results_shard<i>.csv; plot_results.py reads them all.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BUDGETS  # noqa: E402
from train import ARMS, RESULTS, run  # noqa: E402


def done_cells():
    done = set()
    for f in RESULTS.parent.glob("results*.csv"):
        with f.open() as fh:
            done |= {(int(r["budget"]), r["arm"], int(r["seed"])) for r in csv.DictReader(fh)}
    return done


def append_to(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    cells = [(b, a, s) for s in args.seeds for b in args.budgets for a in args.arms]
    priority = lambda c: (c[2] != 0, c[0] > 2, c[0], ARMS.index(c[1]))  # noqa: E731
    cells = sorted(cells, key=priority)
    done = done_cells()
    todo = [c for i, c in enumerate(cells) if c not in done and i % args.nshards == args.shard]
    out = RESULTS.parent / (f"results_shard{args.shard}.csv" if args.nshards > 1 else "results.csv")
    print(f"{len(todo)} cells to run ({len(done)} already done)")
    t0 = time.time()
    for i, (b, a, s) in enumerate(todo, 1):
        row = run(b, a, s, max_steps=args.max_steps)
        append_to(out, row)
        print(f"[{i}/{len(todo)}] budget={b:2d} arm={a:14s} seed={s}  J={row['jmean']:.4f}  es={row['es_iou']:.3f}  best_step={row['best_step']:4d}  {row['minutes']:.1f} min  (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
