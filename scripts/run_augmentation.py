"""Figures for the data-generation half of the README (dissertation ch. 3-4).

    python scripts/run_augmentation.py

Writes to ``figures/`` and prints the numbers quoted in the README.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from ofi import (
    horn_schunck,
    horn_schunck_pyramid,
    make_disc,
    make_pair,
    make_phantom,
    make_texture,
    propagate_semi_lagrangian,
    psnr,
    rigid_flow,
    rotation_flow,
)
from ofi.augment import (
    ROTATION,
    SADDLE,
    SHEAR,
    SQUEEZE,
    area_preserving_perturbation,
    hybrid_evolve,
    linearized_reconstruction,
    localized_family,
    propagate_ode,
    gaussian_window,
    linear_flow,
    scale_flow,
)

OUT = Path(__file__).resolve().parent.parent / "figures"
SHAPE = (160, 160)
ANGLE, DX, DY = 8.0, 4.0, -3.0  # rigid motion: rotation plus translation
PHANTOM = make_phantom(SHAPE, n_blobs=8, seed=4)
(U_TRUE, V_TRUE), FINV = rigid_flow(SHAPE, ANGLE, DX, DY)
I0, I1 = make_pair(PHANTOM, FINV)
U, V = horn_schunck_pyramid(I0, I1)  # everything below uses the *estimated* flow


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def fig_homotopy():
    eps = [0.0, 0.25, 0.5, 0.75, 1.0]
    truth = [make_pair(PHANTOM, rigid_flow(SHAPE, ANGLE * e, DX * e, DY * e)[1])[1] for e in eps]
    lin = [linearized_reconstruction(I0, I1, *scale_flow(U, V, e)) for e in eps]
    tra = [propagate_semi_lagrangian(I0, *scale_flow(U, V, e), 1.0) for e in eps]

    fig, ax = plt.subplots(3, 5, figsize=(13, 8))
    for j, e in enumerate(eps):
        ax[0, j].imshow(truth[j], cmap="gray", vmin=0, vmax=1)
        _clean(ax[0, j], f"truth, eps = {e}")
        ax[1, j].imshow(lin[j], cmap="gray", vmin=0, vmax=1)
        _clean(ax[1, j], f"linearised (3.1): {psnr(lin[j], truth[j]):.1f} dB")
        ax[2, j].imshow(tra[j], cmap="gray", vmin=0, vmax=1)
        _clean(ax[2, j], f"transported: {psnr(tra[j], truth[j]):.1f} dB")
    ax[0, 0].set_ylabel("ground truth", fontsize=10)
    ax[1, 0].set_ylabel("E1 - (Ex*eps*u + Ey*eps*v)", fontsize=10)
    ax[2, 0].set_ylabel("propagate along (eps*u, eps*v)", fontsize=10)
    fig.suptitle("Global homotopy (u, v) -> (eps*u, eps*v), estimated flow. Linearised = blend; transport = motion.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "homotopy.png", dpi=110)
    plt.close(fig)

    # GIF of the transported family, forward then back.
    ts = list(np.linspace(0, 1, 16))
    frames = [propagate_semi_lagrangian(I0, *scale_flow(U, V, e), 1.0) for e in ts]
    frames = frames + frames[-2:0:-1]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
    _clean(ax)
    fig.tight_layout(pad=0)

    def update(k):
        im.set_data(frames[k])
        return (im,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=True)
    anim.save(OUT / "homotopy.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    return {e: (psnr(lin[j], truth[j]), psnr(tra[j], truth[j])) for j, e in enumerate(eps)}


def fig_localized_family():
    members = localized_family(U, V, n=5, width=16.0, rng=11, gain=1.5)
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.8))
    ax[0, 0].imshow(I0, cmap="gray", vmin=0, vmax=1)
    _clean(ax[0, 0], "frame 0")
    ax[1, 0].imshow(np.abs(I1 - I0), cmap="magma", vmin=0, vmax=0.4)
    _clean(ax[1, 0], "|frame 1 - frame 0| (global motion)")
    for k, ((h, kk), u_s, v_s) in enumerate(members, start=1):
        img = propagate_semi_lagrangian(I0, u_s, v_s, 1.0)
        ax[0, k].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[0, k].add_patch(Circle((h, kk), 2 * 16.0, fill=False, ec="tab:red", lw=1.2, ls="--"))
        _clean(ax[0, k], f"member {k}")
        ax[1, k].imshow(np.abs(img - I0), cmap="magma", vmin=0, vmax=0.4)
        ax[1, k].add_patch(Circle((h, kk), 2 * 16.0, fill=False, ec="w", lw=0.8, ls="--"))
        _clean(ax[1, k], f"|member {k} - frame 0|")
    fig.suptitle("Localised family (3.2): the recovered flow gated by a Gaussian window at a random centre (2-sigma circle). Only that region moves.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "localized_family.png", dpi=110)
    plt.close(fig)


def fig_area_preserving():
    gens = [("rotation", ROTATION), ("squeeze", SQUEEZE), ("shear", SHEAR), ("saddle [[1,1],[3,-1]]", SADDLE)]
    center = (SHAPE[1] * 0.42, SHAPE[0] * 0.55)
    width, amp = 18.0, 5.0
    mask = make_disc(SHAPE, center=center, radius=16, ring=0)
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    step = 6
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.8))
    rows = []
    for j, (name, A) in enumerate(gens):
        u, v = area_preserving_perturbation(SHAPE, A, center, width, amplitude=amp)
        ax[0, j].imshow(np.zeros(SHAPE), cmap="gray", vmin=0, vmax=1, alpha=0)
        ax[0, j].quiver(xs[::step, ::step], ys[::step, ::step], u[::step, ::step], -v[::step, ::step], color="tab:blue", scale=60)
        ax[0, j].set_xlim(0, SHAPE[1])
        ax[0, j].set_ylim(SHAPE[0], 0)
        ax[0, j].set_aspect("equal")
        _clean(ax[0, j], f"{name}: windowed div-free field")
        img = propagate_ode(I0, u, v, 1.0, steps=16, method="midpoint")
        moved = propagate_ode(mask, u, v, 1.0, steps=16, method="midpoint")
        ratio = moved.sum() / mask.sum()
        ax[1, j].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[1, j].contour(mask, levels=[0.5], colors="tab:red", linewidths=0.8, linestyles="--")
        ax[1, j].contour(moved, levels=[0.5], colors="tab:orange", linewidths=1.0)
        _clean(ax[1, j], f"applied to frame 0; disc area x{ratio:.4f}")
        # Naive alternative: window the linear field itself (3.2 style), same peak displacement.
        nu, nv = linear_flow(SHAPE, 0.1 * A, 1.0, center=center)
        f = gaussian_window(SHAPE, center, width)
        nu, nv = f * nu, f * nv
        peak = np.hypot(nu, nv).max()
        nu, nv = amp * nu / peak, amp * nv / peak
        naive = propagate_ode(mask, nu, nv, 1.0, steps=16, method="midpoint").sum() / mask.sum()
        rows.append((name, ratio, naive))
    fig.suptitle("Area-preserving perturbations (4.5): traceless generators, localised through a stream function. Red = original disc, orange = moved.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "area_preserving.png", dpi=110)
    plt.close(fig)
    return rows


HYBRID_ANGLE = 20.0  # large enough that a single linearisation is not good enough


def fig_hybrid():
    _, finv = rotation_flow(SHAPE, HYBRID_ANGLE)
    i0, i1 = make_pair(PHANTOM, finv)
    res = hybrid_evolve(i0, i1, n_steps=12, step=0.5)
    res_pyr = hybrid_evolve(i0, i1, n_steps=12, step=0.5, levels=4)
    up, vp = horn_schunck_pyramid(i0, i1, levels=4)
    single = propagate_semi_lagrangian(i0, up, vp, 1.0)
    single_err = float(np.linalg.norm(single - i1) / np.linalg.norm(i1))
    u1, v1, _ = horn_schunck(i0, i1)  # single-scale, one shot: the dissertation's starting point
    one_scale = propagate_semi_lagrangian(i0, u1, v1, 1.0)
    one_scale_err = float(np.linalg.norm(one_scale - i1) / np.linalg.norm(i1))

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.1])
    picks = [0, 1, 2, 4, 8, 12]
    for j, k in enumerate(picks):
        a = fig.add_subplot(gs[0, j])
        a.imshow(res.frames[k], cmap="gray", vmin=0, vmax=1)
        _clean(a, f"step {k}: rel. err {res.rel_error[k]:.3f}")
    a = fig.add_subplot(gs[1, :])
    a.plot(res.rel_error, marker="o", label="hybrid loop, single-scale HS inside (4.4)")
    a.plot(res_pyr.rel_error, marker="s", label="hybrid loop, coarse-to-fine HS inside")
    a.axhline(one_scale_err, ls=":", c="tab:red", label=f"one shot, single-scale HS flow: {one_scale_err:.3f}")
    a.axhline(single_err, ls="--", c="gray", label=f"one shot, coarse-to-fine HS flow: {single_err:.3f}")
    a.set_xlabel("step")
    a.set_ylabel("||evolved - frame 1|| / ||frame 1||")
    a.set_title(f"Forward hybrid propagation ({HYBRID_ANGLE:.0f} deg rotation): relative error to the target frame")
    a.grid(alpha=0.3)
    a.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid.png", dpi=110)
    plt.close(fig)
    return res.rel_error, res_pyr.rel_error, single_err, one_scale_err


def observability():
    """3.3: a flat disc rotating is invisible to the flow; texture makes it observable."""
    shape = (128, 128)
    (ut, vt), finv = rotation_flow(shape, 5.0)
    disc = make_disc(shape, radius=30, ring=0)
    tex = make_texture(shape, sigma=2.0, seed=3)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    cx, cy = (shape[1] - 1) / 2, (shape[0] - 1) / 2
    inside = np.hypot(xs - cx, ys - cy) < 24
    rows = {}
    for name, img in [("flat disc", disc), ("textured disc", disc * (0.7 + 0.3 * tex))]:
        a, b = make_pair(img, finv)
        u, v = horn_schunck_pyramid(a, b)
        rows[name] = float(np.hypot(u - ut, v - vt)[inside].mean())
    return rows, float(np.hypot(ut, vt)[inside].mean())


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()

    h = fig_homotopy()
    print("homotopy.png / homotopy.gif -- PSNR vs. truth (dB)")
    print(f"  {'eps':>5}  {'linearised':>10}  {'transport':>10}")
    for e, (a, b) in h.items():
        print(f"  {e:5.2f}  {a:10.1f}  {b:10.1f}")

    fig_localized_family()
    print("\nlocalized_family.png")

    rows = fig_area_preserving()
    print("\narea_preserving.png -- disc area after perturbation / before")
    print(f"  {'generator':24s} {'stream-fn window':>16s} {'naive f*Ax':>12s}")
    for name, r, nv in rows:
        print(f"  {name:24s} {r:16.4f} {nv:12.4f}")

    errs, errs_p, single, one_scale = fig_hybrid()
    print("\nhybrid.png -- relative error")
    print(f"  one shot single-scale: {one_scale:.3f}   one shot pyramid: {single:.3f}")
    print(f"  hybrid single-scale  step 0: {errs[0]:.3f}  step 2: {errs[2]:.3f}  step 4: {errs[4]:.3f}  step 12: {errs[12]:.3f}")
    print(f"  hybrid pyramid       step 0: {errs_p[0]:.3f}  step 2: {errs_p[2]:.3f}  step 4: {errs_p[4]:.3f}  step 12: {errs_p[12]:.3f}")

    rows, motion = observability()
    print(f"\nobservability (3.3) -- EPE inside a disc rotating 5 deg, mean true motion {motion:.2f} px")
    for name, e in rows.items():
        print(f"  {name:14s} {e:.3f} px")

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
