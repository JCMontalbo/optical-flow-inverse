"""J-mean vs. labelled frames per video, one line per arm, seed bands; plus the pre-registered verdict.

    python experiments/downstream/plot_results.py
"""

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

FIG = Path(__file__).resolve().parents[2] / "figures" / "downstream.png"
LABELS = {"none": "A  no augmentation", "standard": "B  standard (flip/rotate/scale/shear)", "flow": "C  flow-generated pairs",
          "standard+flow": "D  standard + flow", "propagate": "E  labels carried to real neighbours"}


def load():
    cells = defaultdict(list)
    for path in sorted(RESULTS.parent.glob("results*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                cells[(int(r["budget"]), r["arm"])].append(float(r["jmean"]))
    return cells


def main():
    cells = load()
    budgets = sorted({b for b, _ in cells})
    fig, ax = plt.subplots(figsize=(8, 5))
    table = {}
    for arm in ARMS:
        xs, m, lo, hi = [], [], [], []
        for b in budgets:
            v = cells.get((b, arm))
            if not v:
                continue
            xs.append(b)
            m.append(np.mean(v))
            lo.append(np.min(v))
            hi.append(np.max(v))
            table[(b, arm)] = (np.mean(v), np.std(v), len(v), np.min(v), np.max(v))
        if xs:
            ax.plot(xs, m, marker="o", label=LABELS[arm])
            ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("labelled frames per training video (30 videos)")
    ax.set_ylabel("J-mean on DAVIS 2016 val (20 videos)")
    ax.set_title("Does flow-based augmentation help a small U-Net with scarce labels?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)

    print(f"{'budget':>6} " + " ".join(f"{a:>16}" for a in ARMS))
    for b in budgets:
        row = []
        for a in ARMS:
            t = table.get((b, a))
            row.append(f"{t[0]:.3f}+-{t[1]:.3f}(n{t[2]})" if t else "-")
        print(f"{b:>6} " + " ".join(f"{r:>16}" for r in row))

    # pre-registered verdict
    def gap(b, a1, a2):
        t1, t2 = table.get((b, a1)), table.get((b, a2))
        if not t1 or not t2:
            return None
        return (t1[0] - t2[0]) * 100, (t1[3] > t2[4]) or (t2[3] > t1[4])  # points, seed ranges disjoint (either direction)

    print("\nPre-registered criteria (docs/downstream_plan.md):")
    for b in [x for x in (1, 2) if x in budgets]:
        for arm in ("flow", "standard+flow"):
            g = gap(b, arm, "standard")
            if g:
                print(f"  H1  budget {b}: {arm} - standard = {g[0]:+.1f} pts, seed ranges disjoint: {g[1]}  -> {'supported' if g[0] >= 2 and g[1] else ('small effect' if g[0] > 0 else 'not supported')}")
        g = gap(b, "flow", "propagate")
        if g:
            print(f"  H2  budget {b}: flow - propagate = {g[0]:+.1f} pts  -> {'supported' if g[0] >= 1 else 'not supported'}")
    if 20 in budgets and 1 in budgets:
        g1, g20 = gap(1, "flow", "standard"), gap(20, "flow", "standard")
        if g1 and g20:
            print(f"  H3  flow - standard gap: {g1[0]:+.1f} pts at 1 label/video, {g20[0]:+.1f} pts at 20")


if __name__ == "__main__":
    main()
