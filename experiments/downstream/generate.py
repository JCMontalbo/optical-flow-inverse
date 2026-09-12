"""Offline generation of the flow-based training pairs (arms C and E), cached per labelled frame.

For every labelled training frame k (union over all label budgets):

* flow k -> k+4, k-4, k+1, k-1 (grey, coarse-to-fine Horn-Schunck), each accepted only if
  propagating frame k along it predicts the real neighbour at >= --min-psnr dB
  (and better than doing nothing); rejected flows and their pairs are dropped and counted.
* arm E  "propagate": (real neighbour frame, label of k carried along the flow), up to 4 pairs.
* arm C  "flow": 8 generated (image, label) pairs from the accepted flow to the farthest
  neighbour (k+4 preferred, then k-4, k+1, k-1), so the variants differ from the real frame
  by several pixels rather than the ~1 px of adjacent DAVIS frames:
    3.1  homotopy  eps in {0.25, 0.5, 0.75}
    3.2  two Gaussian windows at random centres, only the region under the window moves (gain 3)
    4.5  windowed rotation / squeeze / shear (peak 6% of the height) at the point of largest
         camera-relative motion, added to half the flow
  The image (all three channels) and the label are carried along the same field.

    python experiments/downstream/generate.py            # all labelled frames, 24 processes
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import H, W, all_labelled_frames, load_video, to_gray, videos  # noqa: E402
from ofi import horn_schunck_pyramid, propagate_semi_lagrangian, psnr  # noqa: E402
from ofi.augment import ROTATION, SHEAR, SQUEEZE, area_preserving_perturbation, gaussian_window, propagate_ode  # noqa: E402
from ofi.labels import propagate_label  # noqa: E402

OUT = Path(__file__).resolve().parent / "generated"
MIN_PSNR = 25.0
LEVELS = 5


def move_rgb(frame_u8, u, v, t=1.0, ode=False):
    chans = []
    for c in range(3):
        ch = frame_u8[..., c].astype(np.float32) / 255.0
        if ode:
            out = propagate_ode(ch, u, v, t, steps=6, method="midpoint")
        else:
            out = propagate_semi_lagrangian(ch, u, v, t)
        chans.append(np.clip(out, 0, 1))
    return (np.stack(chans, -1) * 255).round().astype(np.uint8)


def work(job):
    video, k, seed = job
    dest = OUT / f"{video}_{k:05d}.npz"
    if dest.exists():
        return None
    frames, masks = load_video(video)
    gk = to_gray(frames[k])
    rng = np.random.default_rng(seed)
    meta = {"video": video, "k": int(k), "flows": {}}

    flows = {}
    for j in (k + 4, k - 4, k + 1, k - 1):
        if j < 0 or j >= len(frames):
            continue
        gj = to_gray(frames[j])
        u, v = horn_schunck_pyramid(gk, gj, levels=LEVELS)
        p_flow = psnr(propagate_semi_lagrangian(gk, u, v, 1.0), gj)
        p_none = psnr(gk, gj)
        ok = bool(p_flow >= MIN_PSNR and p_flow > p_none)
        meta["flows"][str(j)] = {"psnr_flow": float(p_flow), "psnr_none": float(p_none), "ok": ok, "mean_mag": float(np.hypot(u, v).mean())}
        if ok:
            flows[j] = (u, v)

    # arm E: real neighbour, carried label
    nb_imgs, nb_masks = [], []
    for j, (u, v) in flows.items():
        nb_imgs.append(frames[j])
        nb_masks.append(propagate_label(masks[k], u, v))

    # arm C: 8 generated pairs from the flow to k+1 (fallback k-1)
    imgs, gmasks, kinds = [], [], []
    src = next((flows[j] for j in (k + 4, k - 4, k + 1, k - 1) if j in flows), None)
    meta["source_neighbour"] = next((j for j in (k + 4, k - 4, k + 1, k - 1) if j in flows), None)
    if src is not None:
        u, v = src
        for eps in (0.25, 0.5, 0.75):
            imgs.append(move_rgb(frames[k], eps * u, eps * v))
            gmasks.append(propagate_label(masks[k], eps * u, eps * v))
            kinds.append(f"homotopy_{eps}")
        for _ in range(2):
            c = (rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H)
            f = 3.0 * gaussian_window((H, W), c, 0.14 * H)
            imgs.append(move_rgb(frames[k], f * u, f * v))
            gmasks.append(propagate_label(masks[k], f * u, f * v))
            kinds.append("window")
        rel = np.hypot(u - np.median(u), v - np.median(v))
        from scipy.ndimage import gaussian_filter

        cy, cx = np.unravel_index(np.argmax(gaussian_filter(rel, 0.14 * H)), rel.shape)
        for name, A in (("rotation", ROTATION), ("squeeze", SQUEEZE), ("shear", SHEAR)):
            pu, pv = area_preserving_perturbation((H, W), A, (float(cx), float(cy)), 0.14 * H, amplitude=0.06 * H)
            fu, fv = 0.5 * u + pu, 0.5 * v + pv
            imgs.append(move_rgb(frames[k], fu, fv, ode=True))
            gmasks.append(propagate_label(masks[k], fu, fv))
            kinds.append(name)
    meta["n_generated"] = len(imgs)
    meta["n_neighbours"] = len(nb_imgs)

    OUT.mkdir(exist_ok=True)
    np.savez_compressed(
        dest,
        imgs=np.stack(imgs) if imgs else np.zeros((0, H, W, 3), np.uint8),
        masks=np.stack(gmasks) if gmasks else np.zeros((0, H, W), bool),
        kinds=np.array(kinds),
        nb_imgs=np.stack(nb_imgs) if nb_imgs else np.zeros((0, H, W, 3), np.uint8),
        nb_masks=np.stack(nb_masks) if nb_masks else np.zeros((0, H, W), bool),
        meta=json.dumps(meta),
    )
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="only the first N jobs (smoke test)")
    args = ap.parse_args()
    train = videos("train")
    lab = all_labelled_frames(train)
    jobs = [(v, k, zlib.crc32(f"{v}_{k}".encode())) for v in train for k in sorted(lab[v])]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} labelled frames over {len(train)} training videos")
    t0 = time.time()
    metas = []
    with Pool(args.procs) as pool:
        for i, m in enumerate(pool.imap_unordered(work, jobs)):
            if m is not None:
                metas.append(m)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)")
    # summary statistics over everything on disk
    stats = {"frames": 0, "flows": 0, "flows_ok": 0, "generated": 0, "neighbours": 0, "psnr_flow": [], "psnr_none": []}
    for f in OUT.glob("*.npz"):
        with np.load(f) as z:
            m = json.loads(str(z["meta"]))
        stats["frames"] += 1
        for fl in m["flows"].values():
            stats["flows"] += 1
            stats["flows_ok"] += int(fl["ok"])
            stats["psnr_flow"].append(fl["psnr_flow"])
            stats["psnr_none"].append(fl["psnr_none"])
        stats["generated"] += m["n_generated"]
        stats["neighbours"] += m["n_neighbours"]
    stats["psnr_flow_mean"] = float(np.mean(stats.pop("psnr_flow")))
    stats["psnr_none_mean"] = float(np.mean(stats.pop("psnr_none")))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
