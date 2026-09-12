"""Precompute the flows from every labelled training slice to its neighbours k+-1, k+-2.

Single-scale Horn-Schunck (alpha 0.1, sigma 1) -- the dissertation's own setting, and what works on
MRI: the coarse-to-fine pyramid that helps on video produces garbage on the textureless blood pool.
The field is Gaussian-smoothed (sigma 2 px, as the dissertation does in 4.2 before moving pixels) and
capped at 8 px per pixel; a flow is accepted if propagating slice k along the smoothed field predicts
slice j at least 1 dB better than slice k itself does. One npz per volume: "<k>_<j>" -> float16 [2, H, W], plus meta.

    python experiments/heart/flows.py
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import all_labelled, load_volume, split  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from ofi import horn_schunck, propagate_semi_lagrangian, psnr  # noqa: E402

OUT = Path(__file__).resolve().parent / "flows"
ALPHA, SIGMA, MIN_GAIN = 0.1, 1.0, 1.0
SMOOTH, CAP = 2.0, 8.0
OFFSETS = (1, -1, 2, -2)


def work(vid):
    dest = OUT / f"{vid}.npz"
    if dest.exists():
        return vid, None
    img, _, _ = load_volume(vid)
    ks = sorted(all_labelled([vid])[vid])
    arrays, meta = {}, {}
    for k in ks:
        for d in OFFSETS:
            j = k + d
            if j < 0 or j >= len(img):
                continue
            u, v, _ = horn_schunck(img[k], img[j], alpha=ALPHA, sigma=SIGMA)
            u, v = gaussian_filter(u, SMOOTH), gaussian_filter(v, SMOOTH)
            mag = np.hypot(u, v)
            scale = np.minimum(1.0, CAP / np.maximum(mag, 1e-9))
            u, v = u * scale, v * scale
            p_flow = psnr(propagate_semi_lagrangian(img[k], u, v, 1.0), img[j])
            p_none = psnr(img[k], img[j])
            ok = bool(p_flow >= p_none + MIN_GAIN)
            meta[f"{k}_{j}"] = dict(psnr_flow=float(p_flow), psnr_none=float(p_none), ok=ok, mean_mag=float(np.hypot(u, v).mean()))
            if ok:
                arrays[f"{k}_{j}"] = np.stack([u, v]).astype(np.float16)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(dest, meta=json.dumps(meta), **arrays)
    return vid, meta


def main():
    train, _ = split()
    t0 = time.time()
    stats = dict(flows=0, ok=0, gain=[])
    with Pool(min(14, len(train))) as pool:
        for vid, meta in pool.imap_unordered(work, train):
            if meta is None:
                with np.load(OUT / f"{vid}.npz") as z:
                    meta = json.loads(str(z["meta"]))
            for m in meta.values():
                stats["flows"] += 1
                stats["ok"] += int(m["ok"])
                stats["gain"].append(m["psnr_flow"] - m["psnr_none"])
            print(f"  {vid}: {sum(int(m['ok']) for m in meta.values())}/{len(meta)} flows accepted ({time.time() - t0:.0f}s)", flush=True)
    stats["mean_gain_dB"] = float(np.mean(stats.pop("gain")))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
