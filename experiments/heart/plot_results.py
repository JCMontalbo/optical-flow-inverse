"""Dice vs labelled slices per volume, one line per arm; pre-registered verdict (docs/downstream_heart_plan.md)."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ARMS, RESULTS  # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "figures" / "downstream_heart.png"
LABELS = {"none": "A  no augmentation", "plausible": "B  plausible affine (rot 10, scale, shift)", "elastic": "C  random elastic deformation",
          "flow": "D  recovered-flow family (on the fly)", "propagate": "E  labels carried to neighbour slices"}


def load():
    cells = defaultdict(list)
    for path in sorted(RESULTS.parent.glob("results*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                cells[(int(r["budget"]), r["arm"])].append(float(r["dice"]))
    return cells


def main():
    cells = load()
    budgets = sorted({b for b, _ in cells})
    xl = {b: ("all" if b >= 999 else str(b)) for b in budgets}
    fig, ax = plt.subplots(figsize=(8, 5))
    table = {}
    for arm in ARMS:
        xs, m, lo, hi = [], [], [], []
        for i, b in enumerate(budgets):
            v = cells.get((b, arm))
            if not v:
                continue
            xs.append(i)
            m.append(np.mean(v))
            lo.append(np.min(v))
            hi.append(np.max(v))
            table[(b, arm)] = (np.mean(v), np.std(v), len(v), np.min(v), np.max(v))
        if xs:
            ax.plot(xs, m, marker="o", label=LABELS[arm])
            ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([xl[b] for b in budgets])
    ax.set_xlabel("labelled slices per training volume (14 volumes)")
    ax.set_ylabel("left-atrium Dice, 6 val volumes (3D)")
    ax.set_title("Cardiac MRI, left atrium: augmentation from observed motion vs. the alternatives")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)

    print(f"{'budget':>6} " + " ".join(f"{a:>18}" for a in ARMS))
    for b in budgets:
        print(f"{xl[b]:>6} " + " ".join((f"{table[(b, a)][0]:.3f}+-{table[(b, a)][1]:.3f}(n{table[(b, a)][2]})" if (b, a) in table else "-").rjust(18) for a in ARMS))

    def gap(b, a1, a2):
        t1, t2 = table.get((b, a1)), table.get((b, a2))
        if not t1 or not t2:
            return None
        return (t1[0] - t2[0]) * 100, (t1[3] > t2[4]) or (t2[3] > t1[4])

    print("\nPre-registered criteria (docs/downstream_heart_plan.md):")
    for b in [x for x in (1, 2) if x in budgets]:
        for name, other, thr in (("H1", "plausible", 2), ("H2", "elastic", 2), ("H3", "propagate", 1)):
            g = gap(b, "flow", other)
            if g:
                ok = g[0] >= thr and (g[1] if thr == 2 else True)
                print(f"  {name}  budget {b}: flow - {other} = {g[0]:+.1f} pts, seed ranges disjoint: {g[1]}  -> {'supported' if ok else ('small effect' if g[0] > 0 else 'not supported')}")
    if 999 in budgets:
        for other in ("plausible", "elastic"):
            g = gap(999, "flow", other)
            if g:
                print(f"  H4  all labels: flow - {other} = {g[0]:+.1f} pts")


if __name__ == "__main__":
    main()
