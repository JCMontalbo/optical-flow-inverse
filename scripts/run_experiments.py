"""Regenerate every figure and number quoted in the README.

    python scripts/run_experiments.py

Writes PNGs to ``figures/`` and prints a results table to stdout.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from ofi import (
    endpoint_error,
    horn_schunck,
    horn_schunck_pyramid,
    make_pair,
    make_texture,
    propagate_semi_lagrangian,
    propagate_upwind,
    psnr,
    rotation_flow,
    translation_flow,
)

OUT = Path(__file__).resolve().parent.parent / "figures"
SHAPE = (128, 128)
TEX = make_texture(SHAPE, sigma=3.0, seed=0)


def flow_to_rgb(u, v, vmax=None):
    """Hue = direction, saturation = magnitude (Middlebury-style colour coding)."""
    mag = np.hypot(u, v)
    vmax = vmax or max(float(mag.max()), 1e-9)
    hue = (np.arctan2(-v, -u) + np.pi) / (2 * np.pi)
    hsv = np.stack([hue, np.clip(mag / vmax, 0, 1), np.ones_like(mag)], axis=-1)
    return hsv_to_rgb(hsv)


def fig_flow_recovery():
    (ut, vt), finv = rotation_flow(SHAPE, 2.5)
    i0, i1 = make_pair(TEX, finv)
    u, v = horn_schunck_pyramid(i0, i1)
    vmax = float(np.hypot(ut, vt).max())
    err = np.hypot(u - ut, v - vt)
    epe = endpoint_error(u, v, ut, vt)

    fig, ax = plt.subplots(1, 5, figsize=(15, 3.2))
    ax[0].imshow(i0, cmap="gray")
    ax[0].set_title("frame 0")
    ax[1].imshow(i1, cmap="gray")
    ax[1].set_title("frame 1 (rotated 2.5 deg)")
    ax[2].imshow(flow_to_rgb(ut, vt, vmax))
    ax[2].set_title("true flow")
    ax[3].imshow(flow_to_rgb(u, v, vmax))
    ax[3].set_title("recovered flow")
    im = ax[4].imshow(err, cmap="magma", vmin=0, vmax=0.5)
    ax[4].set_title(f"|error|  (EPE {epe:.3f} px)")
    fig.colorbar(im, ax=ax[4], fraction=0.046)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "flow_recovery.png", dpi=110)
    plt.close(fig)
    return epe


def fig_alpha_sweep():
    alphas = np.logspace(-2.5, 0.5, 13)
    cases = {
        "translation 0.5 px, clean": (translation_flow(SHAPE, 0.4, -0.3), 0.0),
        "translation 0.5 px, noise sd 0.02": (translation_flow(SHAPE, 0.4, -0.3), 0.02),
        "rotation 2 deg, clean": (rotation_flow(SHAPE, 2.0), 0.0),
        "rotation 2 deg, noise sd 0.02": (rotation_flow(SHAPE, 2.0), 0.02),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    for label, (((ut, vt), finv), noise) in cases.items():
        i0, i1 = make_pair(TEX, finv, noise_std=noise)
        epe = []
        for a in alphas:
            u, v, _ = horn_schunck(i0, i1, alpha=a, n_iter=5000)
            epe.append(endpoint_error(u, v, ut, vt))
        ax.semilogx(alphas, epe, marker="o", ms=4, label=label)
        best = int(np.argmin(epe))
        rows.append((label, alphas[best], epe[best]))
    ax.set_xlabel("regularisation weight alpha")
    ax.set_ylabel("endpoint error (px)")
    ax.set_title("Bias-variance trade-off of the smoothness prior")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_sweep.png", dpi=110)
    plt.close(fig)
    return rows


def fig_pyramid_vs_single():
    disps = [0.5, 1, 2, 3, 4, 6, 8]
    single, pyr3, pyr4 = [], [], []
    for d in disps:
        (ut, vt), finv = translation_flow(SHAPE, d * 0.8, d * 0.6)  # |flow| = d
        i0, i1 = make_pair(TEX, finv)
        u, v, _ = horn_schunck(i0, i1)
        single.append(endpoint_error(u, v, ut, vt))
        u, v = horn_schunck_pyramid(i0, i1, levels=3)
        pyr3.append(endpoint_error(u, v, ut, vt))
        u, v = horn_schunck_pyramid(i0, i1, levels=4)
        pyr4.append(endpoint_error(u, v, ut, vt))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(disps, single, marker="o", label="single scale")
    ax.plot(disps, pyr3, marker="s", label="pyramid, 3 levels")
    ax.plot(disps, pyr4, marker="^", label="pyramid, 4 levels")
    ax.plot(disps, disps, ls="--", c="gray", lw=1, label="error = motion (no information)")
    ax.set_xlabel("true displacement (px)")
    ax.set_ylabel("endpoint error (px)")
    ax.set_title("Linearised brightness constancy breaks past ~2 px; coarse-to-fine fixes it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pyramid_vs_single.png", dpi=110)
    plt.close(fig)
    return list(zip(disps, single, pyr3, pyr4))


def fig_convergence():
    (ut, vt), finv = translation_flow(SHAPE, 0.4, -0.3)
    i0, i1 = make_pair(TEX, finv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in [0.03, 0.1, 0.3, 1.0]:
        _, _, h = horn_schunck(i0, i1, alpha=a, n_iter=5000, tol=1e-7)
        ax.semilogy(h, label=f"alpha = {a}")
    ax.set_xlabel("Jacobi iteration")
    ax.set_ylabel("mean update (px)")
    ax.set_title("Convergence slows as alpha grows: the Laplacian couples pixels more strongly")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=110)
    plt.close(fig)


def fig_interpolation():
    """Forward propagation vs. linear blend for frame interpolation."""
    (ut, vt), finv = rotation_flow(SHAPE, 4.0)
    i0, i1 = make_pair(TEX, finv)
    # Ground-truth intermediate frames: rotate by t * angle.
    ts = np.linspace(0, 1, 11)
    truth = [make_pair(TEX, rotation_flow(SHAPE, 4.0 * t)[1])[1] for t in ts]
    u, v = horn_schunck_pyramid(i0, i1)  # estimated, not true, flow

    curves = {
        "linear blend": [],
        "upwind (true flow)": [],
        "semi-Lagrangian (true flow)": [],
        "semi-Lagrangian (estimated flow)": [],
    }
    for t, gt in zip(ts, truth):
        curves["linear blend"].append(psnr((1 - t) * i0 + t * i1, gt))
        curves["upwind (true flow)"].append(psnr(propagate_upwind(i0, ut, vt, t), gt))
        curves["semi-Lagrangian (true flow)"].append(psnr(propagate_semi_lagrangian(i0, ut, vt, t), gt))
        curves["semi-Lagrangian (estimated flow)"].append(psnr(propagate_semi_lagrangian(i0, u, v, t), gt))

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1])
    mid = 5
    panels = [
        ("true frame at t = 0.5", truth[mid]),
        ("linear blend", 0.5 * (i0 + i1)),
        ("upwind, true flow", propagate_upwind(i0, ut, vt, 0.5)),
        ("semi-Lagrangian, true flow", propagate_semi_lagrangian(i0, ut, vt, 0.5)),
        ("semi-Lagrangian, est. flow", propagate_semi_lagrangian(i0, u, v, 0.5)),
    ]
    for k, (title, img) in enumerate(panels):
        a = fig.add_subplot(gs[0, k])
        a.imshow(img[32:96, 32:96], cmap="gray", vmin=0, vmax=1)  # zoom on the centre
        p = "" if k == 0 else f"\n{psnr(img, truth[mid]):.1f} dB"
        a.set_title(title + p, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
    a = fig.add_subplot(gs[1, :])
    for label, ys in curves.items():
        ys = [min(y, 60) for y in ys]
        a.plot(ts, ys, marker="o", ms=4, label=label)
    a.set_xlabel("time t between frames")
    a.set_ylabel("PSNR vs. true frame (dB, capped at 60)")
    a.set_title("Frame interpolation quality: propagating along the flow vs. blending")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "interpolation.png", dpi=110)
    plt.close(fig)
    return {k: v[mid] for k, v in curves.items()}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    epe = fig_flow_recovery()
    print(f"flow_recovery.png — rotation 2.5 deg, pyramid EPE = {epe:.3f} px")

    rows = fig_alpha_sweep()
    print("\nalpha_sweep.png — best alpha per case")
    for label, a, e in rows:
        print(f"  {label:36s} alpha*={a:.3f}  EPE={e:.3f} px")

    rows = fig_pyramid_vs_single()
    print("\npyramid_vs_single.png — EPE (px)")
    print(f"  {'|flow|':>6}  {'single':>7}  {'pyr3':>7}  {'pyr4':>7}")
    for d, s, p3, p4 in rows:
        print(f"  {d:6.1f}  {s:7.3f}  {p3:7.3f}  {p4:7.3f}")

    fig_convergence()
    print("\nconvergence.png")

    mid = fig_interpolation()
    print("\ninterpolation.png — PSNR at t = 0.5 (dB)")
    for k, v in mid.items():
        print(f"  {k:36s} {v:6.1f}")

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
