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
