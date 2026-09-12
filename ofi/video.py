"""Video I/O and tracking helpers shared by the video scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter, zoom

from .augment import gaussian_window


def read_frames(path: Path, start: int, end: int) -> np.ndarray:
    """Grey frames ``[start, end)`` of a video in [0, 1]. Needs ``imageio[ffmpeg]``."""
    import imageio.v2 as iio

    reader = iio.get_reader(str(path), "ffmpeg")
    frames = []
    for k, f in enumerate(reader):
        if k >= end:
            break
        if k >= start:
            frames.append(np.asarray(f)[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114]) / 255.0)
    reader.close()
    return np.stack(frames)


def crop_letterbox(frames: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    rows = np.where(frames.mean(axis=(0, 2)) > thresh)[0]
    cols = np.where(frames.mean(axis=(0, 1)) > thresh)[0]
    return frames[:, rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def resize_frames(frames: np.ndarray, width: int) -> np.ndarray:
    scale = width / frames.shape[2]
    return np.clip(np.stack([zoom(f, scale, order=1) for f in frames]), 0, 1)


def flow_to_rgb(u, v, vmax):
    mag = np.hypot(u, v)
    hue = (np.arctan2(-v, -u) + np.pi) / (2 * np.pi)
    return hsv_to_rgb(np.stack([hue, np.clip(mag / vmax, 0, 1), np.ones_like(mag)], -1))


class WindowTracker:
    """A Gaussian window that follows the subject's motion *relative to the camera*.

    Placed at the maximum of the smoothed camera-relative motion on the first
    pair, then per step advected by the mean flow under it and pulled toward
    the centroid of camera-relative motion under it (mean-shift), so it stays
    on moving pixels instead of drifting onto static background.
    """

    def __init__(self, shape, width: float):
        self.shape = shape
        self.width = width
        self.center = None

    def window(self, u, v):
        H, W = self.shape
        rel = np.hypot(u - np.median(u), v - np.median(v))
        if self.center is None:
            sm = gaussian_filter(rel, self.width)
            cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
            self.center = np.array([cx, cy], dtype=float)
        f = gaussian_window((H, W), tuple(self.center), self.width)
        center = self.center.copy()

        wsum = f.sum()
        nxt = self.center + np.array([(u * f).sum() / wsum, (v * f).sum() / wsum])
        w = f * rel
        if w.sum() > 1e-9:
            ys, xs = np.mgrid[0:H, 0:W]
            centroid = np.array([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()])
            nxt = 0.5 * nxt + 0.5 * centroid
        self.center = np.clip(nxt, [0, 0], [W - 1, H - 1])
        return f, center


def cached_flows(frames, pairs, cache: Path | None, levels: int = 4, alpha: float = 0.1, log=print):
    """Flows for the given frame-index pairs, read from / written to ``cache`` (npz) when given."""
    import time

    from .horn_schunck import horn_schunck_pyramid

    key = lambda a, b: f"{a}_{b}"  # noqa: E731
    store = {}
    if cache is not None and Path(cache).exists():
        with np.load(cache) as z:
            store = {k: z[k] for k in z.files}
    out, todo = {}, []
    for a, b in pairs:
        if key(a, b) in store:
            out[(a, b)] = (store[key(a, b)][0], store[key(a, b)][1])
        else:
            todo.append((a, b))
    t0 = time.time()
    for i, (a, b) in enumerate(todo):
        u, v = horn_schunck_pyramid(frames[a], frames[b], alpha=alpha, levels=levels)
        out[(a, b)] = (u, v)
        store[key(a, b)] = np.stack([u, v])
        if i % 10 == 0:
            log(f"  flow {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
    if cache is not None and todo:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **store)
    return out


def synthesize_family(frames, flows1, variants, window: float, rho: float, cap: float):
    """Synthetic clips with persistently modified motion (see scripts/synthesize_video.py).

    For each variant the deviation ``(du, dv)`` from the real motion is
    accumulated frame to frame -- advected along the real flow, decayed by
    ``rho``, capped at ``cap`` px -- and each synthetic frame is the real frame
    resampled once along the current deviation. Returns per-variant lists of
    frames and of deviation fields, plus the tracked window centres.
    """
    from .augment import area_preserving_perturbation
    from .propagate import propagate_semi_lagrangian
    from .synthetic import warp

    n, H, W = frames.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    tracker = WindowTracker((H, W), window)
    devs = [(np.zeros((H, W)), np.zeros((H, W))) for _ in variants]
    syn = [[] for _ in variants]
    dev_hist = [[] for _ in variants]
    centers = []
    for k in range(n - 1):
        u, v = flows1[(k, k + 1)]
        f, center = tracker.window(u, v)
        centers.append(center)
        for i, spec in enumerate(variants):
            du, dv = devs[i]
            syn[i].append(propagate_semi_lagrangian(frames[k], du, dv, 1.0))
            dev_hist[i].append((du, dv))
            if spec["kind"] == "amplify":
                eu, ev = (spec["gain"] - 1) * f * u, (spec["gain"] - 1) * f * v
            else:
                a = spec["amp"] * np.sin(2 * np.pi * k / spec["period"])
                eu, ev = area_preserving_perturbation((H, W), spec["A"], tuple(center), window, amplitude=abs(a))
                eu, ev = np.sign(a) * eu, np.sign(a) * ev
            du = rho * warp(du, xs - u, ys - v, order=1) + eu
            dv = rho * warp(dv, xs - u, ys - v, order=1) + ev
            m = np.hypot(du, dv)
            s = np.minimum(1.0, cap / np.maximum(m, 1e-9))
            devs[i] = (du * s, dv * s)
    return syn, dev_hist, centers


def default_variants(gain: float, amp: float):
    """The synthetic-clip recipes used in the README: what happens to the motion under the tracked window."""
    from .augment import ROTATION, SHEAR, SQUEEZE

    return [
        dict(name=f"region moves {gain:g}x", kind="amplify", gain=gain),
        dict(name="region frozen", kind="amplify", gain=0.0),
        dict(name="pulsing swirl", kind="perturb", A=ROTATION, amp=amp, period=16),
        dict(name="pulsing squeeze", kind="perturb", A=SQUEEZE, amp=amp, period=20),
        dict(name="pulsing shear", kind="perturb", A=SHEAR, amp=amp, period=24),
    ]


def silhouette_label(frame: np.ndarray, thresh: float = 0.35, prev: np.ndarray | None = None) -> np.ndarray:
    """Pseudo-label for a dark subject on a light background.

    The dark components (holes filled); the largest one, or, when ``prev`` is
    given, the one overlapping the previous frame's label most -- so the label
    stays on the same subject when another dark object enters the frame.
    """
    from scipy import ndimage as ndi

    m = ndi.binary_fill_holes(frame < thresh)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    if prev is not None:
        score = ndi.sum(prev.astype(float), lab, range(1, n + 1))
        if np.max(score) > 0:
            return lab == (int(np.argmax(score)) + 1)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)
