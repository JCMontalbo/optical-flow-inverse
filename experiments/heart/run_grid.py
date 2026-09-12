"""Run the heart grid, sharded; cells that decide the criteria first (seed 0, budgets 1 and 2).

    python experiments/heart/run_grid.py --shard 0 --nshards 3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import BUDGETS  # noqa: E402
from train import ARMS, RESULTS, append_to, run  # noqa: E402


def done_cells():
    done = set()
    for f in RESULTS.parent.glob("results*.csv"):
        with f.open() as fh:
            done |= {(int(r["budget"]), r["arm"], int(r["seed"])) for r in csv.DictReader(fh)}
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    cells = sorted([(b, a, s) for s in args.seeds for b in args.budgets for a in args.arms], key=lambda c: (c[2] != 0, c[0] > 2, c[0], ARMS.index(c[1])))
    done = done_cells()
    todo = [c for i, c in enumerate(cells) if c not in done and i % args.nshards == args.shard]
    out = RESULTS.parent / (f"results_shard{args.shard}.csv" if args.nshards > 1 else "results.csv")
    print(f"{len(todo)} cells to run ({len(done)} done)", flush=True)
    t0 = time.time()
    for i, (b, a, s) in enumerate(todo, 1):
        row = run(b, a, s, steps=args.steps)
        append_to(out, row)
        print(f"[{i}/{len(todo)}] budget={b:3d} arm={a:10s} seed={s}  Dice={row['dice']:.4f}  es={row['es_dice']:.3f}  step={row['best_step']:4d}  flows_ok={row['n_flows_ok']}  {row['minutes']:.1f} min  (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
