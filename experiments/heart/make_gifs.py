"""Two GIFs for the README, on a validation patient the segmenter never saw.

heart_propagation.gif -- one labelled slice in the middle of the volume; its label carried slice by slice
                         outward in both directions along the recovered flow (cyan) against the true
                         label of every slice (red), with IoU.
heart_augmentation.gif -- one labelled slice; every frame is a fresh random member of each augmentation
                          arm: plausible affine, random elastic, the recovered-flow family.

    python experiments/heart/make_gifs.py
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
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from data import load_volume, split  # noqa: E402
from flowaug import elastic, flow_family, plausible_affine  # noqa: E402

from ofi import horn_schunck  # noqa: E402
from ofi.labels import chain_label, iou  # noqa: E402

FIG = HERE.parents[1] / "figures"
DPI = 100


def smooth_flow(a, b):
    u, v, _ = horn_schunck(a, b, alpha=0.1, sigma=1.0)
    u, v = gaussian_filter(u, 2.0), gaussian_filter(v, 2.0)
    m = np.hypot(u, v)
    s = np.minimum(1.0, 8.0 / np.maximum(m, 1e-9))
    return u * s, v * s


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def propagation_gif(vid):
    img, m, _ = load_volume(vid)
    have = [k for k in range(len(m)) if m[k].any()]
    k0 = have[len(have) // 2]
    # carry the label outward, chaining one slice at a time
    carried = {k0: m[k0]}
    for direction in (1, -1):
        prev = k0
        k = k0 + direction
        while 0 <= k < len(img):
            u, v = smooth_flow(img[prev], img[k])
            carried[k] = chain_label(carried[prev], [(u, v)])
            prev, k = k, k + direction
    # play it the way it was computed: hold on the labelled slice, sweep up, come back, sweep down
    # ... and stop four slices past where the atrium ends: after that there is nothing left to compare against
    top, bot = min(len(img) - 1, have[-1] + 4), max(0, have[0] - 4)
    up, down = list(range(k0, top + 1)), list(range(k0, bot - 1, -1))
    frames = [k0] * 4 + up + [up[-1]] * 3 + [k0] * 3 + down + [down[-1]] * 3

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.9))
    im0 = ax[0].imshow(img[frames[0]], cmap="gray", vmin=0, vmax=1)
    im1 = ax[1].imshow(img[frames[0]], cmap="gray", vmin=0, vmax=1)
    _clean(ax[0])
    _clean(ax[1])
    fig.text(0.5, 0.965, f"Validation patient {vid}: label drawn on ONE mid-volume slice (#{k0}), carried outward both ways along the recovered flow (to 4 slices past the atrium)", ha="center", fontsize=10)
    conts = []
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(i):
        k = frames[i]
        for c in conts:
            c.remove()
        conts.clear()
        im0.set_data(img[k])
        im1.set_data(img[k])
        if m[k].any():
            conts.append(ax[0].contour(m[k], levels=[0.5], colors="r", linewidths=1.2))
        conts.append(ax[1].contour(carried[k], levels=[0.5], colors="c", linewidths=1.2))
        if m[k].any():
            conts.append(ax[1].contour(m[k], levels=[0.5], colors="r", linewidths=0.7, linestyles="--"))
        sc = iou(carried[k], m[k]) if (m[k].any() or carried[k].any()) else 1.0
        ax[0].set_title(f"slice {k}: true label", fontsize=10)
        arrow = "up" if k > k0 else "down"
        ax[1].set_title(f"carried {abs(k - k0)} slices {arrow} from #{k0}: IoU {sc:.2f}" if k != k0 else f"the one labelled slice (#{k0}, mid-volume)", fontsize=10)
        return [im0, im1]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 8, blit=False)
    anim.save(FIG / "heart_propagation.gif", writer=PillowWriter(fps=8), dpi=DPI)
    plt.close(fig)
    ious = [iou(carried[k], m[k]) for k in frames if m[k].any()]
    dists = [abs(k - k0) for k in frames if m[k].any()]
    return k0, ious, dists


def augmentation_gif(vid, k, n_frames=36):
    img, m, _ = load_volume(vid)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(img[k]).to(dev).view(1, 1, *img[k].shape)
    y = torch.from_numpy(m[k]).float().to(dev).view(1, 1, *m[k].shape)
    ys, xs = np.where(m[k])
    cen = torch.tensor([[xs.mean(), ys.mean()]], dtype=torch.float32, device=dev)
    # a recovered flow to a neighbour, as used in training
    u, v = smooth_flow(img[k], img[k + 2])
    flow = torch.from_numpy(np.stack([u, v])).float().to(dev).unsqueeze(0)
    gen = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(n_frames):
        a = plausible_affine(x, y, gen)
        e = elastic(x, y, gen)
        f = flow_family(x, y, flow, cen, gen)
        samples.append([(t[0][0, 0].cpu().numpy(), t[1][0, 0].cpu().numpy()) for t in (a, e, f)])

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.3))
    ims = [ax[0].imshow(img[k], cmap="gray", vmin=0, vmax=1)]
    ax[0].contour(m[k], levels=[0.5], colors="r", linewidths=1.2)
    _clean(ax[0], "the one labelled slice")
    titles = ["plausible affine (small rotation / shift)", "random elastic deformation", "recovered-flow family (the dissertation)"]
    for j in range(3):
        ims.append(ax[j + 1].imshow(samples[0][j][0], cmap="gray", vmin=0, vmax=1))
        _clean(ax[j + 1], titles[j])
    conts = []
    fig.suptitle("What the network sees each step: a fresh sample from each augmentation arm (label carried along, cyan)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(i):
        for c in conts:
            c.remove()
        conts.clear()
        for j in range(3):
            ims[j + 1].set_data(samples[i][j][0])
            conts.append(ax[j + 1].contour(samples[i][j][1], levels=[0.5], colors="c", linewidths=1.2))
        return ims

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / 4, blit=False)
    anim.save(FIG / "heart_augmentation.gif", writer=PillowWriter(fps=4), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    _, val = split()
    vid = val[0]
    k0, ious, dists = propagation_gif(vid)
    print(f"{vid}: label carried from slice {k0}; IoU vs true label: mean {np.mean(ious):.3f}, at 5 slices {np.mean([i for i, d in zip(ious, dists) if d == 5]):.3f}, at 10 {np.mean([i for i, d in zip(ious, dists) if d == 10]):.3f}, at 20 {np.mean([i for i, d in zip(ious, dists) if d == 20]):.3f}")
    if "--propagation-only" not in sys.argv:
        augmentation_gif(vid, k0)
    print("wrote figures/heart_propagation.gif" + ("" if "--propagation-only" in sys.argv else ", figures/heart_augmentation.gif"))
