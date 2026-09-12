"""Run the framework on a video as frames stream in.

    python scripts/augment_video.py clip.mp4 --start 0 --end 58 --out figures/video

For every consecutive pair (k, k+1) the flow is estimated, a Gaussian window
is advected along that flow so it follows a moving feature, and two synthetic
frames are produced from the *real* frame k:

    localised   -- the motion under the window is amplified by --gain
                   (everything else moves as in the video)      [3.2]
    perturbed   -- the flow plus a localised area-preserving swirl
                   of --amplitude px at the window                [4.5]

Each synthetic frame is derived from the real frame at that instant, so
errors do not accumulate; pass --accumulate to chain them instead
(synthetic_{k+1} from synthetic_k), which is the dissertation's evolution
setting and drifts the way Fig. 4.21-4.22 describe.

Writes ``stream.gif`` (original | flow | localised | perturbed), a summary
PNG, and the per-pair reconstruction PSNR so you can see how well the
estimated flow explains the real next frame.

Reading MP4 needs ``imageio`` with ``imageio-ffmpeg`` (``pip install imageio[ffmpeg]``).
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
from ofi.augment import ROTATION, area_preserving_perturbation, propagate_ode
from ofi.video import WindowTracker, crop_letterbox, flow_to_rgb, read_frames, resize_frames


def _clean(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="video file; downloaded to this path from --url if missing")
    ap.add_argument("--url", default=None, help="download the video from here if the file does not exist")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=60)
    ap.add_argument("--width", type=int, default=320, help="resize frames to this width")
    ap.add_argument("--out", type=Path, default=Path("out_video"))
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--window", type=float, default=None, help="window std-dev in px (default 12%% of height)")
    ap.add_argument("--gain", type=float, default=3.0, help="motion multiplier under the window")
    ap.add_argument("--amplitude", type=float, default=None, help="swirl peak displacement in px (default 4%% of height)")
    ap.add_argument("--accumulate", action="store_true", help="chain synthetic frames instead of deriving each from the real frame")
    ap.add_argument("--gif-stride", type=int, default=1, help="write every n-th pair to the GIF")
    ap.add_argument("--credit", default="", help="attribution line printed under the GIF")
    ap.add_argument("--save-flows", action="store_true", help="also write flows.npz (large)")
    args = ap.parse_args()

    if not args.video.exists():
        if not args.url:
            raise SystemExit(f"{args.video} not found and no --url given")
        args.video.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.url} -> {args.video}")
        urllib.request.urlretrieve(args.url, args.video)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    frames = resize_frames(crop_letterbox(read_frames(args.video, args.start, args.end)), args.width)
    n, H, W = frames.shape
    window = args.window or 0.12 * H
    amp = args.amplitude or 0.04 * H
    print(f"{n} frames of {H}x{W} from {args.video.name} [{args.start}:{args.end}]")

    # --- stream through the pairs
    t0 = time.time()
    flows, centers, psnrs, loc_frames, per_frames = [], [], [], [], []
    tracker = WindowTracker((H, W), window)
    syn_loc = frames[0].copy()
    syn_per = frames[0].copy()
    for k in range(n - 1):
        a, b = frames[k], frames[k + 1]
        u, v = horn_schunck_pyramid(a, b, alpha=args.alpha, levels=args.levels)
        flows.append((u, v))
        psnrs.append((psnr(a, b), psnr(propagate_semi_lagrangian(a, u, v, 1.0), b)))

        mag = np.hypot(u, v)
        f, center = tracker.window(u, v)
        centers.append(center)

        src_loc = syn_loc if args.accumulate else a
        src_per = syn_per if args.accumulate else a
        # 3.2: amplify only the motion under the window
        syn_loc = propagate_semi_lagrangian(src_loc, u * (1 + (args.gain - 1) * f), v * (1 + (args.gain - 1) * f), 1.0)
        # 4.5: the real flow plus a localised, area-preserving swirl
        pu, pv = area_preserving_perturbation((H, W), ROTATION, tuple(center), window, amplitude=amp)
        syn_per = propagate_ode(src_per, u + pu, v + pv, 1.0, steps=6, method="midpoint")
        loc_frames.append(syn_loc)
        per_frames.append(syn_per)

        if k % 10 == 0:
            print(f"  pair {k:3d}: |flow| mean {mag.mean():.2f} max {mag.max():.1f} px; PSNR next frame {psnrs[-1][0]:.1f} -> {psnrs[-1][1]:.1f} dB  ({time.time() - t0:.0f}s)")

    psnrs = np.array(psnrs)
    vmax = float(np.percentile(np.hypot(*np.array(flows).transpose(1, 0, 2, 3)), 99))

    # --- GIF: original | flow | localised | perturbed
    idx = list(range(0, n - 1, args.gif_stride))
    fig, ax = plt.subplots(2, 2, figsize=(2 * 4.2, 2 * 4.2 * H / W + 0.9))
    ax = ax.ravel()
    ims = [
        ax[0].imshow(frames[0], cmap="gray", vmin=0, vmax=1),
        ax[1].imshow(flow_to_rgb(*flows[0], vmax)),
        ax[2].imshow(loc_frames[0], cmap="gray", vmin=0, vmax=1),
        ax[3].imshow(per_frames[0], cmap="gray", vmin=0, vmax=1),
    ]
    circles = [Circle(tuple(centers[0]), 2 * window, fill=False, ec="tab:red", lw=1.0, ls="--") for _ in range(3)]
    for a_, c in zip([ax[0], ax[2], ax[3]], circles):
        a_.add_patch(c)
    titles = ["video frame k", "estimated flow k -> k+1", f"motion under the window x{args.gain:g} (3.2)", f"flow + local swirl, {amp:.0f} px (4.5)"]
    for a_, t in zip(ax, titles):
        _clean(a_, t)
    if args.credit:
        fig.text(0.5, 0.005, args.credit, ha="center", va="bottom", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    def update(j):
        k = idx[j]
        ims[0].set_data(frames[k])
        ims[1].set_data(flow_to_rgb(*flows[k], vmax))
        ims[2].set_data(loc_frames[k])
        ims[3].set_data(per_frames[k])
        for c in circles:
            c.center = tuple(centers[k])
        return ims + circles

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / 12, blit=True)
    anim.save(out / "stream.gif", writer=PillowWriter(fps=12 // args.gif_stride), dpi=64)
    plt.close(fig)

    # --- summary figure: one pair in detail + PSNR trace
    k = int(np.argmax(psnrs[:, 1] - psnrs[:, 0]))
    fig = plt.figure(figsize=(13, 3.2 + 13 / 4 * H / W), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[13 / 4 * H / W, 2.6])
    panels = [
        (frames[k], f"frame {k}", "gray"),
        (flow_to_rgb(*flows[k], vmax), "estimated flow", None),
        (loc_frames[k], f"window motion x{args.gain:g}", "gray"),
        (per_frames[k], f"flow + local swirl", "gray"),
    ]
    for j, (img, t, cm) in enumerate(panels):
        a_ = fig.add_subplot(gs[0, j])
        a_.imshow(img, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None)
        if j != 1:
            a_.add_patch(Circle(tuple(centers[k]), 2 * window, fill=False, ec="tab:red", lw=1.0, ls="--"))
        _clean(a_, t)
    a_ = fig.add_subplot(gs[1, :])
    a_.plot(psnrs[:, 0], label="PSNR(frame k, frame k+1): do nothing")
    a_.plot(psnrs[:, 1], label="PSNR(frame k moved along estimated flow, frame k+1)")
    a_.set_xlabel("pair k")
    a_.set_ylabel("dB")
    a_.set_title("How well the estimated flow explains the real next frame")
    a_.grid(alpha=0.3)
    a_.legend(fontsize=9)
    fig.savefig(out / "stream_summary.png", dpi=110)
    plt.close(fig)

    np.savetxt(out / "psnr.csv", psnrs, delimiter=",", header="psnr_do_nothing_dB,psnr_along_flow_dB", comments="")
    if args.save_flows:
        np.savez_compressed(out / "flows.npz", u=np.array([f[0] for f in flows]), v=np.array([f[1] for f in flows]), centers=np.array(centers))
    print(f"\nmean PSNR to the next real frame: do nothing {psnrs[:, 0].mean():.1f} dB, along the flow {psnrs[:, 1].mean():.1f} dB")
    print(f"wrote {out}/stream.gif, stream_summary.png, psnr.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
