"""Create new video data from a real clip, two ways.

    python scripts/synthesize_video.py clip.mp4 --start 0 --end 58 --out figures/video

A. New frames that never existed, validated (dissertation goal 2)
   Every odd frame is held out. It is generated from its even neighbours by
   propagating frame k along the flow k -> k+2 for half the time, and compared
   with the real held-out frame. Linear blending is the baseline. The same
   machinery then produces --factor x slow motion of a short segment
   (``slowmo.gif``: real frames held vs. blended vs. propagated).

B. New videos of the same scene (dissertation goal 1)
   A family of synthetic clips in which the motion is modified persistently:
   a tracked region moving --gain x faster, a frozen region, a pulsing swirl,
   squeeze or shear. The *deviation* from the real motion is accumulated
   frame to frame (advected along the flow, with memory --rho), and each
   synthetic frame is a single resampling of the real frame at that instant,
   so the clips diverge visibly from the original without accumulating blur.
   ``family.gif`` plays the original and the variants side by side;
   ``family_stats.png`` shows how far each variant is from the original and
   that its sharpness stays that of the real footage.

Needs ``imageio[ffmpeg]`` to read MP4.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from ofi import horn_schunck_pyramid, propagate_semi_lagrangian, psnr
from ofi.augment import ROTATION, SHEAR, SQUEEZE, area_preserving_perturbation
from ofi.synthetic import warp
from ofi.video import WindowTracker, crop_letterbox, read_frames, resize_frames


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def sharpness(img: np.ndarray) -> float:
    gy, gx = np.gradient(img)
    return float(np.sqrt(gx**2 + gy**2).mean())


def estimate_flows(frames, pairs, levels, alpha, label):
    t0 = time.time()
    out = {}
    for i, (a, b) in enumerate(pairs):
        out[(a, b)] = horn_schunck_pyramid(frames[a], frames[b], alpha=alpha, levels=levels)
        if i % 10 == 0:
            print(f"  {label}: {i + 1}/{len(pairs)} flows ({time.time() - t0:.0f}s)")
    return out


# --------------------------------------------------------------------------- A


def held_out_validation(frames, flows2, out, credit):
    n = len(frames)
    rows = []
    for k in range(0, n - 2, 2):
        u, v = flows2[(k, k + 2)]
        real = frames[k + 1]
        prop = propagate_semi_lagrangian(frames[k], u, v, 0.5)
        blend = 0.5 * (frames[k] + frames[k + 2])
        hold = frames[k]
        rows.append((k + 1, psnr(hold, real), psnr(blend, real), psnr(prop, real)))
    rows = np.array(rows)
    np.savetxt(out / "heldout_psnr.csv", rows, delimiter=",", header="held_out_frame,psnr_nearest_dB,psnr_blend_dB,psnr_propagated_dB", comments="", fmt="%.3f")

    # Detail on the frame where propagation gains most over blending.
    j = int(np.argmax(rows[:, 3] - rows[:, 2]))
    k = int(rows[j, 0]) - 1
    u, v = flows2[(k, k + 2)]
    prop = propagate_semi_lagrangian(frames[k], u, v, 0.5)
    blend = 0.5 * (frames[k] + frames[k + 2])
    real = frames[k + 1]
    H, W = real.shape
    # zoom on the region of largest motion
    mag = np.hypot(u - np.median(u), v - np.median(v))
    cy, cx = np.unravel_index(np.argmax(mag), mag.shape)
    r = int(0.3 * H)
    y0, x0 = max(0, min(cy - r, H - 2 * r)), max(0, min(cx - r, W - 2 * r))
    crop = (slice(y0, y0 + 2 * r), slice(x0, x0 + 2 * r))

    fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])
    panels = [(real, f"real held-out frame {k + 1}"), (blend, f"blend of {k} and {k + 2}: {psnr(blend, real):.1f} dB"),
              (prop, f"frame {k} propagated to t=0.5: {psnr(prop, real):.1f} dB"), (np.abs(prop - real), "|propagated - real|")]
    for j2, (img, t) in enumerate(panels):
        a = fig.add_subplot(gs[0, j2])
        a.imshow(img, cmap="gray" if j2 < 3 else "magma", vmin=0, vmax=1 if j2 < 3 else 0.3)
        a.add_patch(plt.Rectangle((x0, y0), 2 * r, 2 * r, fill=False, ec="tab:red", lw=1.0, ls="--"))
        _clean(a, t)
        a2 = fig.add_subplot(gs[1, j2])
        a2.imshow(img[crop], cmap="gray" if j2 < 3 else "magma", vmin=0, vmax=1 if j2 < 3 else 0.3, interpolation="nearest")
        _clean(a2, "zoom")
    a = fig.add_subplot(gs[2, :])
    a.plot(rows[:, 0], rows[:, 1], label="nearest real frame (frame k)", c="gray")
    a.plot(rows[:, 0], rows[:, 2], label="linear blend of frames k and k+2")
    a.plot(rows[:, 0], rows[:, 3], label="frame k propagated along flow k -> k+2 to t = 0.5")
    a.set_xlabel("held-out frame")
    a.set_ylabel("PSNR vs. the real held-out frame (dB)")
    a.set_title(f"Generating held-out frames: mean {rows[:, 3].mean():.1f} dB propagated vs {rows[:, 2].mean():.1f} dB blended")
    a.grid(alpha=0.3)
    a.legend(fontsize=9)
    if credit:
        fig.text(0.99, 0.005, credit, ha="right", va="bottom", fontsize=7, color="gray")
    fig.savefig(out / "heldout.png", dpi=110)
    plt.close(fig)
    return rows


def slow_motion(frames, flows1, out, factor, seg, credit):
    a0, a1 = seg
    ts = np.arange(factor) / factor
    real, blend, prop = [], [], []
    for k in range(a0, a1):
        u, v = flows1[(k, k + 1)]
        for t in ts:
            real.append(frames[k])
            blend.append((1 - t) * frames[k] + t * frames[k + 1])
            prop.append(propagate_semi_lagrangian(frames[k], u, v, float(t)))
    H, W = frames[0].shape
    fig, ax = plt.subplots(1, 3, figsize=(3 * 4.0, 4.0 * H / W + 0.7))
    ims = [ax[0].imshow(real[0], cmap="gray", vmin=0, vmax=1), ax[1].imshow(blend[0], cmap="gray", vmin=0, vmax=1), ax[2].imshow(prop[0], cmap="gray", vmin=0, vmax=1)]
    for a, t in zip(ax, [f"real frames, each held {factor}x", "linear blend (ghosting)", f"propagated along the flow ({factor}x new frames)"]):
        _clean(a, t)
    if credit:
        fig.text(0.5, 0.005, credit, ha="center", va="bottom", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    def update(j):
        ims[0].set_data(real[j])
        ims[1].set_data(blend[j])
        ims[2].set_data(prop[j])
        return ims

    anim = FuncAnimation(fig, update, frames=len(real), interval=1000 / 12, blit=True)
    anim.save(out / "slowmo.gif", writer=PillowWriter(fps=12), dpi=64)
    plt.close(fig)


# --------------------------------------------------------------------------- B


def make_variants(rng, gain, amp, H):
    """Per-variant recipe: what is done to the motion under the tracked window."""
    return [
        dict(name=f"region moves {gain:g}x", kind="amplify", gain=gain),
        dict(name="region frozen", kind="amplify", gain=0.0),
        dict(name=f"pulsing swirl", kind="perturb", A=ROTATION, amp=amp, period=16),
        dict(name=f"pulsing squeeze", kind="perturb", A=SQUEEZE, amp=amp, period=20),
        dict(name=f"pulsing shear", kind="perturb", A=SHEAR, amp=amp, period=24),
    ]


def synthesize_family(frames, flows1, variants, window, rho, cap, out, gif_stride, credit):
    n, H, W = frames.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    tracker = WindowTracker((H, W), window)
    devs = [(np.zeros((H, W)), np.zeros((H, W))) for _ in variants]
    syn = [[] for _ in variants]
    centers = []
    for k in range(n - 1):
        u, v = flows1[(k, k + 1)]
        f, center = tracker.window(u, v)
        centers.append(center)
        for i, spec in enumerate(variants):
            du, dv = devs[i]
            # the synthetic frame: the real frame, resampled once along the accumulated deviation
            syn[i].append(propagate_semi_lagrangian(frames[k], du, dv, 1.0))
            # per-frame deviation from the real motion under the window
            if spec["kind"] == "amplify":
                eu, ev = (spec["gain"] - 1) * f * u, (spec["gain"] - 1) * f * v
            else:
                a = spec["amp"] * np.sin(2 * np.pi * k / spec["period"])
                eu, ev = area_preserving_perturbation((H, W), spec["A"], tuple(center), window, amplitude=abs(a))
                eu, ev = np.sign(a) * eu, np.sign(a) * ev
            # advect the accumulated deviation with the real motion so it stays on the content,
            # decay it with memory rho, add this frame's contribution, and cap its magnitude
            du = rho * warp(du, xs - u, ys - v, order=1) + eu
            dv = rho * warp(dv, xs - u, ys - v, order=1) + ev
            m = np.hypot(du, dv)
            s = np.minimum(1.0, cap / np.maximum(m, 1e-9))
            devs[i] = (du * s, dv * s)

    # stats: distance from the original, and sharpness relative to the original
    dist = np.array([[psnr(syn[i][k], frames[k]) for k in range(n - 1)] for i in range(len(variants))])
    sharp = np.array([[sharpness(syn[i][k]) / sharpness(frames[k]) for k in range(n - 1)] for i in range(len(variants))])

    # GIF: original + variants
    idx = list(range(0, n - 1, gif_stride))
    cols = 3
    rowsn = int(np.ceil((len(variants) + 1) / cols))
    fig, ax = plt.subplots(rowsn, cols, figsize=(cols * 4.0, rowsn * (4.0 * H / W + 0.45) + 0.3))
    ax = ax.ravel()
    ims = [ax[0].imshow(frames[0], cmap="gray", vmin=0, vmax=1)]
    circ = [Circle(tuple(centers[0]), 2 * window, fill=False, ec="tab:red", lw=0.9, ls="--")]
    ax[0].add_patch(circ[0])
    _clean(ax[0], "original")
    for i, spec in enumerate(variants, start=1):
        ims.append(ax[i].imshow(syn[i - 1][0], cmap="gray", vmin=0, vmax=1))
        c = Circle(tuple(centers[0]), 2 * window, fill=False, ec="tab:red", lw=0.9, ls="--")
        ax[i].add_patch(c)
        circ.append(c)
        _clean(ax[i], spec["name"])
    for a in ax[len(variants) + 1 :]:
        a.axis("off")
    if credit:
        fig.text(0.5, 0.005, credit, ha="center", va="bottom", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    def update(j):
        k = idx[j]
        ims[0].set_data(frames[k])
        for i in range(len(variants)):
            ims[i + 1].set_data(syn[i][k])
        for c in circ:
            c.center = tuple(centers[k])
        return ims + circ

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / 12, blit=True)
    anim.save(out / "family.gif", writer=PillowWriter(fps=12 // gif_stride), dpi=64)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
    for i, spec in enumerate(variants):
        ax[0].plot(dist[i], label=spec["name"])
        ax[1].plot(sharp[i], label=spec["name"])
    ax[0].set_xlabel("frame")
    ax[0].set_ylabel("PSNR(variant, original) dB")
    ax[0].set_title("How different each synthetic clip is from the original (lower = more different)")
    ax[0].set_ylim(15, 60)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("sharpness ratio to original")
    ax[1].set_title("Sharpness stays that of the real footage (one resampling per frame)")
    ax[1].axhline(1.0, c="gray", ls="--", lw=0.8)
    ax[1].set_ylim(0.8, 1.1)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.savefig(out / "family_stats.png", dpi=110)
    plt.close(fig)
    return dist, sharp


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
    ap.add_argument("--factor", type=int, default=4, help="slow-motion factor")
    ap.add_argument("--slowmo-pairs", type=int, default=12, help="how many real pairs go into slowmo.gif")
    ap.add_argument("--window", type=float, default=None, help="window std-dev in px (default 14%% of height)")
    ap.add_argument("--gain", type=float, default=3.0)
    ap.add_argument("--amplitude", type=float, default=None, help="pulse peak per frame in px (default 1.5%% of height)")
    ap.add_argument("--rho", type=float, default=0.9, help="memory of the accumulated deviation")
    ap.add_argument("--cap", type=float, default=None, help="max accumulated deviation in px (default 10%% of height)")
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
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
    print(f"{n} frames of {H}x{W}")

    flows1 = estimate_flows(frames, [(k, k + 1) for k in range(n - 1)], args.levels, args.alpha, "consecutive")
    flows2 = estimate_flows(frames, [(k, k + 2) for k in range(0, n - 2, 2)], args.levels, args.alpha, "skip-one")

    rows = held_out_validation(frames, flows2, out, args.credit)
    print(f"\nA. held-out frames ({len(rows)}): PSNR nearest {rows[:, 1].mean():.1f}  blend {rows[:, 2].mean():.1f}  propagated {rows[:, 3].mean():.1f} dB")
    best = int(np.argmax([np.hypot(*flows1[(k, k + 1)]).mean() for k in range(n - 1 - args.slowmo_pairs)]))
    slow_motion(frames, flows1, out, args.factor, (best, best + args.slowmo_pairs), args.credit)
    print(f"   slowmo.gif: pairs {best}-{best + args.slowmo_pairs}, {args.factor}x")

    rng = np.random.default_rng(args.seed)
    variants = make_variants(rng, args.gain, amp, H)
    dist, sharp = synthesize_family(frames, flows1, variants, window, args.rho, cap, out, args.gif_stride, args.credit)
    print("\nB. synthetic family (per variant: PSNR to original at last frame, mean sharpness ratio)")
    for spec, d, sh in zip(variants, dist, sharp):
        print(f"   {spec['name']:22s} {d[-1]:5.1f} dB   {sh.mean():.3f}")
    print(f"\nwrote {out}/heldout.png heldout_psnr.csv slowmo.gif family.gif family_stats.png")


if __name__ == "__main__":
    main()
