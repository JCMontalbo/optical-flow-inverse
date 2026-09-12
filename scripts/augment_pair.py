"""Generate a family of new images from two frames of your own.

    python scripts/augment_pair.py frame0.png frame1.png --out out/

Takes any two successive frames (PNG/JPG; colour is converted to grey),
recovers the flow between them, and writes:

    out/flow.png              recovered flow (colour-coded) and the frame pair
    out/homotopy.png          the ε-family from frame 0 to frame 1 (3.1)
    out/homotopy.gif          the same as an animation
    out/localized.png         n members with motion gated to a random window (3.2)
    out/area_preserving.png   frame 0 under localised rotation / squeeze / shear / saddle (4.5)
    out/flow.npz              u, v arrays for your own experiments

Frames are downscaled so the longer side is at most --max-size pixels;
Horn-Schunck in NumPy is O(pixels x iterations) and 512 px takes ~10 s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle
from scipy.ndimage import zoom

from ofi import horn_schunck_pyramid, propagate_semi_lagrangian
from ofi.augment import (
    ROTATION,
    SADDLE,
    SHEAR,
    SQUEEZE,
    area_preserving_perturbation,
    localized_family,
    propagate_ode,
    scale_flow,
)


def load_gray(path: Path, max_size: int) -> np.ndarray:
    img = mpimg.imread(path).astype(float)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    if img.max() > 1.0:
        img = img / 255.0
    scale = max_size / max(img.shape)
    if scale < 1:
        img = zoom(img, scale, order=1)
    img = (img - img.min()) / max(img.max() - img.min(), 1e-9)
    return img


def flow_to_rgb(u, v):
    mag = np.hypot(u, v)
    hue = (np.arctan2(-v, -u) + np.pi) / (2 * np.pi)
    sat = np.clip(mag / max(float(np.percentile(mag, 99)), 1e-9), 0, 1)
    return hsv_to_rgb(np.stack([hue, sat, np.ones_like(mag)], -1))


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame0", type=Path)
    ap.add_argument("frame1", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--max-size", type=int, default=384)
    ap.add_argument("--levels", type=int, default=4, help="pyramid levels for the flow estimate")
    ap.add_argument("--alpha", type=float, default=0.1, help="Horn-Schunck regularisation (images are in [0,1])")
    ap.add_argument("--n", type=int, default=5, help="members of the localised family")
    ap.add_argument("--width", type=float, default=None, help="window std-dev in px (default: 10%% of the short side)")
    ap.add_argument("--gain", type=float, default=1.5, help="multiplier on the flow inside each window")
    ap.add_argument("--amplitude", type=float, default=None, help="peak displacement (px) of area-preserving perturbations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    i0 = load_gray(args.frame0, args.max_size)
    i1 = load_gray(args.frame1, args.max_size)
    if i0.shape != i1.shape:
        raise SystemExit(f"frames differ in shape after loading: {i0.shape} vs {i1.shape}")
    short = min(i0.shape)
    width = args.width or 0.10 * short
    amp = args.amplitude or 0.03 * short

    print(f"frames {i0.shape}; estimating flow (levels={args.levels}, alpha={args.alpha}) ...")
    u, v = horn_schunck_pyramid(i0, i1, alpha=args.alpha, levels=args.levels)
    mag = np.hypot(u, v)
    print(f"  mean |flow| {mag.mean():.2f} px, 99th pct {np.percentile(mag, 99):.2f} px")
    np.savez(out / "flow.npz", u=u, v=v)

    # --- flow figure
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    ax[0].imshow(i0, cmap="gray", vmin=0, vmax=1)
    _clean(ax[0], "frame 0")
    ax[1].imshow(i1, cmap="gray", vmin=0, vmax=1)
    _clean(ax[1], "frame 1")
    ax[2].imshow(flow_to_rgb(u, v))
    _clean(ax[2], "recovered flow (hue = direction)")
    rec = propagate_semi_lagrangian(i0, u, v, 1.0)
    err = float(np.linalg.norm(rec - i1) / np.linalg.norm(i1))
    ax[3].imshow(rec, cmap="gray", vmin=0, vmax=1)
    _clean(ax[3], f"frame 0 moved along the flow (rel. err {err:.3f})")
    fig.tight_layout()
    fig.savefig(out / "flow.png", dpi=110)
    plt.close(fig)

    # --- 3.1 homotopy
    eps = np.linspace(0, 1, 6)
    fig, ax = plt.subplots(1, len(eps), figsize=(2.6 * len(eps), 2.9))
    for a, e in zip(ax, eps):
        a.imshow(propagate_semi_lagrangian(i0, *scale_flow(u, v, e), 1.0), cmap="gray", vmin=0, vmax=1)
        _clean(a, f"eps = {e:.1f}")
    fig.suptitle("Global homotopy (u, v) -> (eps u, eps v)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "homotopy.png", dpi=110)
    plt.close(fig)

    ts = list(np.linspace(0, 1, 20))
    frames = [propagate_semi_lagrangian(i0, *scale_flow(u, v, e), 1.0) for e in ts]
    frames = frames + frames[-2:0:-1]
    fig, ax = plt.subplots(figsize=(4, 4 * i0.shape[0] / i0.shape[1]))
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
    _clean(ax)
    fig.tight_layout(pad=0)
    anim = FuncAnimation(fig, lambda k: (im.set_data(frames[k]), im)[1:], frames=len(frames), interval=70, blit=True)
    anim.save(out / "homotopy.gif", writer=PillowWriter(fps=14))
    plt.close(fig)

    # --- 3.2 localised family
    members = localized_family(u, v, n=args.n, width=width, rng=args.seed, gain=args.gain)
    fig, ax = plt.subplots(2, args.n + 1, figsize=(2.6 * (args.n + 1), 5.4))
    ax[0, 0].imshow(i0, cmap="gray", vmin=0, vmax=1)
    _clean(ax[0, 0], "frame 0")
    ax[1, 0].imshow(np.abs(i1 - i0), cmap="magma", vmin=0, vmax=0.4)
    _clean(ax[1, 0], "|frame 1 - frame 0|")
    for k, ((h, kk), u_s, v_s) in enumerate(members, start=1):
        img = propagate_semi_lagrangian(i0, u_s, v_s, 1.0)
        ax[0, k].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[0, k].add_patch(Circle((h, kk), 2 * width, fill=False, ec="tab:red", lw=1.0, ls="--"))
        _clean(ax[0, k], f"member {k}")
        ax[1, k].imshow(np.abs(img - i0), cmap="magma", vmin=0, vmax=0.4)
        _clean(ax[1, k], f"|member {k} - frame 0|")
        mpimg.imsave(out / f"member_{k}.png", img, cmap="gray", vmin=0, vmax=1)
    fig.suptitle(f"Localised family: motion gated to a Gaussian window (std {width:.0f} px, gain {args.gain})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "localized.png", dpi=110)
    plt.close(fig)

    # --- 4.5 area-preserving perturbations at the point of largest motion
    ky, kx = np.unravel_index(np.argmax(zoom(mag, 0.25, order=1)), zoom(mag, 0.25, order=1).shape)
    center = (kx * 4, ky * 4)
    gens = [("rotation", ROTATION), ("squeeze", SQUEEZE), ("shear", SHEAR), ("saddle", SADDLE)]
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.4))
    ax[0].imshow(i0, cmap="gray", vmin=0, vmax=1)
    ax[0].add_patch(Circle(center, 2 * width, fill=False, ec="tab:red", lw=1.0, ls="--"))
    _clean(ax[0], "frame 0 (window at max motion)")
    for a, (name, A) in zip(ax[1:], gens):
        pu, pv = area_preserving_perturbation(i0.shape, A, center, width, amplitude=amp)
        img = propagate_ode(i0, u + pu, v + pv, 1.0, steps=8, method="midpoint")
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        _clean(a, f"flow + local {name} ({amp:.0f} px)")
        mpimg.imsave(out / f"perturbed_{name}.png", img, cmap="gray", vmin=0, vmax=1)
    fig.suptitle("Recovered flow plus a localised area-preserving perturbation, then propagated", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "area_preserving.png", dpi=110)
    plt.close(fig)

    print(f"wrote {out}/: flow.png homotopy.png homotopy.gif localized.png area_preserving.png member_*.png perturbed_*.png flow.npz")


if __name__ == "__main__":
    main()
