"""Carry a segmentation label along the flow -- through real video and onto synthetic data.

    python scripts/propagate_labels.py clip.mp4 --start 0 --end 58 --out figures/video

The clip has no ground-truth masks, so a pseudo-label is derived per frame
(the largest dark component: the character's silhouette on snow). Three
experiments, all scored against those per-frame pseudo-labels with IoU:

  1. one step   -- label frame k, propagate along flow k -> k+1, compare with the
                   label of frame k+1. Baseline: leave the label where it is.
  2. sparse     -- label every N-th frame only and propagate in between (soft
                   memberships, hardened once). Baseline: hold the last labelled
                   frame's mask. N = 2 ... 20, and N = "all" (frame 0 only).
  3. synthetic  -- for each synthetic clip of scripts/synthesize_video.py, carry
                   the real frame's label along the *same* deviation field the
                   image was resampled with, and check it against the silhouette
                   re-extracted from the synthetic image. Baseline: the unmoved
                   label. This is what makes generated frames usable as training
                   data: the (image, label) pair stays consistent.

Writes ``labels.gif``, ``labels.png`` and ``labels_iou.csv``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ofi.labels import chain_label, iou, propagate_label
from ofi.video import (
    cached_flows,
    crop_letterbox,
    default_variants,
    read_frames,
    resize_frames,
    silhouette_label,
    synthesize_family,
)


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--url", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=58)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("out_video"))
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--thresh", type=float, default=0.35, help="intensity threshold for the silhouette pseudo-label")
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--amplitude", type=float, default=None)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--credit", default="")
    args = ap.parse_args()

    if not args.video.exists():
        if not args.url:
            raise SystemExit(f"{args.video} not found and no --url given")
        args.video.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(args.url, args.video)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    frames = resize_frames(crop_letterbox(read_frames(args.video, args.start, args.end)), args.width)
    n, H, W = frames.shape
    window = args.window or 0.14 * H
    amp = args.amplitude or 0.015 * H
    cap = args.cap or 0.10 * H
    labels = [silhouette_label(frames[0], args.thresh)]
    for f in frames[1:]:
        labels.append(silhouette_label(f, args.thresh, prev=labels[-1]))
    labels = np.stack(labels)
    flows = cached_flows(frames, [(k, k + 1) for k in range(n - 1)], out / "flows_cache.npz", args.levels, args.alpha)
    print(f"{n} frames of {H}x{W}; pseudo-label covers {labels.mean() * 100:.0f}% of the frame on average")

    # 1. one step
    one_step, one_step_base = [], []
    for k in range(n - 1):
        u, v = flows[(k, k + 1)]
        one_step.append(iou(propagate_label(labels[k], u, v), labels[k + 1]))
        one_step_base.append(iou(labels[k], labels[k + 1]))
    one_step, one_step_base = np.array(one_step), np.array(one_step_base)

    # 2. sparse anchors: label every N-th frame, propagate to the frames in between
    intervals = [2, 3, 5, 10, 20, n]
    sparse, sparse_base = [], []
    for N in intervals:
        prop, hold = [], []
        for k in range(n):
            a = (k // N) * N
            if k == a:
                continue
            moved = chain_label(labels[a], [flows[(j, j + 1)] for j in range(a, k)])
            prop.append(iou(moved, labels[k]))
            hold.append(iou(labels[a], labels[k]))
        sparse.append(np.mean(prop))
        sparse_base.append(np.mean(hold))
    sparse, sparse_base = np.array(sparse), np.array(sparse_base)
    # for the GIF: the frame-0 label chained through the whole shot
    chain_frames = [labels[0]]
    for k in range(n - 1):
        chain_frames.append(chain_label(labels[0], [flows[(j, j + 1)] for j in range(0, k + 1)]))

    # 3. synthetic clips carry their labels
    variants = default_variants(args.gain, amp)
    syn, devs, centers = synthesize_family(frames, flows, variants, window, args.rho, cap)
    syn_labels = [[propagate_label(labels[k], *devs[i][k]) for k in range(n - 1)] for i in range(len(variants))]
    consist = np.array([[iou(syn_labels[i][k], silhouette_label(syn[i][k], args.thresh, prev=labels[k])) for k in range(n - 1)] for i in range(len(variants))])
    consist_base = np.array([[iou(labels[k], silhouette_label(syn[i][k], args.thresh, prev=labels[k])) for k in range(n - 1)] for i in range(len(variants))])

    rows = np.column_stack([np.arange(n - 1), one_step_base, one_step] + [consist_base[i] for i in range(len(variants))] + [consist[i] for i in range(len(variants))])
    hdr = "pair,one_step_static,one_step_propagated," + ",".join(f"{v['name'].replace(' ', '_')}_static" for v in variants) + "," + ",".join(f"{v['name'].replace(' ', '_')}_propagated" for v in variants)
    np.savetxt(out / "labels_iou.csv", rows, delimiter=",", header=hdr, comments="", fmt="%.4f")
    np.savetxt(out / "labels_sparse_iou.csv", np.column_stack([intervals, sparse_base, sparse]), delimiter=",", header="label_every_n_frames,hold_last_label,propagated", comments="", fmt="%.4f")

    print(f"\n1. one step   IoU: propagated {one_step.mean():.3f}  vs static {one_step_base.mean():.3f}")
    print("2. sparse     label every N frames, IoU on the unlabelled frames: propagated vs hold last label")
    for N, p_, h_ in zip(intervals, sparse, sparse_base):
        print(f"   N = {('all' if N == n else N):>3}: {p_:.3f} vs {h_:.3f}")
    print("3. synthetic  IoU(label carried along, silhouette of synthetic frame) vs unmoved label, mean over frames:")
    for i, spec in enumerate(variants):
        print(f"   {spec['name']:20s} {consist[i].mean():.3f}  vs  {consist_base[i].mean():.3f}")

    # --- figure
    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    a = fig.add_subplot(gs[0, 0])
    a.plot(one_step, label="propagated one step")
    a.plot(one_step_base, label="label left in place", c="gray")
    a.set_title("1. label frame k -> frame k+1")
    a.set_xlabel("pair k")
    a.set_ylabel("IoU with frame k+1 pseudo-label")
    a.set_ylim(0, 1)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    a = fig.add_subplot(gs[0, 1])
    xl = [str(N) if N != n else "frame 0 only" for N in intervals]
    a.plot(xl, sparse, marker="o", label="propagated from the last labelled frame")
    a.plot(xl, sparse_base, marker="s", c="gray", label="last labelled frame's mask held")
    a.set_title("2. label every N-th frame only")
    a.set_xlabel("N (frames between labelled frames)")
    a.set_ylabel("IoU on the unlabelled frames")
    a.set_ylim(0, 1)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    a = fig.add_subplot(gs[0, 2])
    x = np.arange(len(variants))
    a.bar(x - 0.2, consist_base.mean(1), 0.4, label="unmoved label", color="gray")
    a.bar(x + 0.2, consist.mean(1), 0.4, label="label carried along the deviation")
    a.set_xticks(x)
    a.set_xticklabels([v["name"] for v in variants], rotation=20, fontsize=8, ha="right")
    a.set_ylim(0, 1)
    a.set_title("3. synthetic clips keep their labels")
    a.set_ylabel("IoU, mean over frames")
    a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=8)
    fig.savefig(out / "labels.png", dpi=110)
    plt.close(fig)

    # --- GIF: real frame with per-frame label (red) and chained label (cyan); three variants with carried labels
    idx = list(range(0, n - 1, args.gif_stride))
    show = [0, 2, 3]  # region moves 3x, pulsing swirl, pulsing squeeze
    fig, ax = plt.subplots(2, 2, figsize=(2 * 4.2, 2 * 4.2 * H / W + 0.9))
    ax = ax.ravel()
    ims = [ax[0].imshow(frames[0], cmap="gray", vmin=0, vmax=1)]
    _clean(ax[0], "video: per-frame label (red), label propagated from frame 0 (cyan)")
    for j, i in enumerate(show, start=1):
        ims.append(ax[j].imshow(syn[i][0], cmap="gray", vmin=0, vmax=1))
        _clean(ax[j], f"{variants[i]['name']}: label carried along (cyan)")
    conts = []
    if args.credit:
        fig.text(0.5, 0.005, args.credit, ha="center", va="bottom", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    def update(t):
        k = idx[t]
        for c in conts:
            c.remove()
        conts.clear()
        ims[0].set_data(frames[k])
        conts.append(ax[0].contour(labels[k], levels=[0.5], colors="r", linewidths=0.9))
        conts.append(ax[0].contour(chain_frames[k], levels=[0.5], colors="c", linewidths=0.9))
        for j, i in enumerate(show, start=1):
            ims[j].set_data(syn[i][k])
            conts.append(ax[j].contour(syn_labels[i][k], levels=[0.5], colors="c", linewidths=0.9))
        return ims

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / 12, blit=False)
    anim.save(out / "labels.gif", writer=PillowWriter(fps=12 // args.gif_stride), dpi=64)
    plt.close(fig)
    print(f"\nwrote {out}/labels.gif labels.png labels_iou.csv")


if __name__ == "__main__":
    main()
