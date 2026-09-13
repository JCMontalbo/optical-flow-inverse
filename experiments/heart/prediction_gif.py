"""Sweep a held-out patient with three segmenters trained on ONE labelled slice per patient (seed 0).

    python experiments/heart/prediction_gif.py

Writes figures/heart_predictions.gif: truth | plausible affine | random elastic | recovered-flow family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load_volume, split  # noqa: E402
from train import run, tens  # noqa: E402

FIG = HERE.parents[1] / "figures"
ARMS = [("plausible", "plausible affine", "tab:orange"), ("elastic", "random elastic", "tab:green"), ("flow", "recovered-flow family", "c")]


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 1.0


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def main():
    _, val = split()
    vid = val[0]
    img, m, _ = load_volume(vid)
    preds, vol_dice = {}, {}
    for arm, _, _ in ARMS:
        row = run(1, arm, 0, keep_model=True)
        model = row["model"].eval()
        with torch.no_grad():
            x = tens(img, "cuda").unsqueeze(1)
            p = torch.cat([(model(x[i : i + 32]) > 0).squeeze(1).cpu() for i in range(0, len(x), 32)]).numpy()
        preds[arm] = p
        vol_dice[arm] = dice(p, m)
        print(f"{arm:10s} val Dice (6 patients) {row['dice']:.3f}   this patient {vol_dice[arm]:.3f}")
        del model
        torch.cuda.empty_cache()

    frames = list(range(len(img)))
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
    ims = [a.imshow(img[0], cmap="gray", vmin=0, vmax=1) for a in ax]
    for a in ax:
        _clean(a)
    fig.suptitle(f"Held-out patient {vid}: segmenters trained on ONE labelled slice per patient (14 patients), swept through the volume", fontsize=10)
    conts = []
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(i):
        k = frames[i]
        for c in conts:
            c.remove()
        conts.clear()
        for im in ims:
            im.set_data(img[k])
        if m[k].any():
            conts.append(ax[0].contour(m[k], levels=[0.5], colors="r", linewidths=1.3))
        ax[0].set_title(f"slice {k}: truth", fontsize=10)
        for j, (arm, name, col) in enumerate(ARMS, start=1):
            if preds[arm][k].any():
                conts.append(ax[j].contour(preds[arm][k], levels=[0.5], colors=col, linewidths=1.3))
            if m[k].any():
                conts.append(ax[j].contour(m[k], levels=[0.5], colors="r", linewidths=0.6, linestyles="--"))
            d = dice(preds[arm][k], m[k])
            ax[j].set_title(f"{name}: slice Dice {d:.2f}  (3D {vol_dice[arm]:.2f})", fontsize=10)
        return ims

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 8, blit=False)
    anim.save(FIG / "heart_predictions.gif", writer=PillowWriter(fps=8), dpi=100)
    plt.close(fig)
    print("wrote figures/heart_predictions.gif")


if __name__ == "__main__":
    main()
