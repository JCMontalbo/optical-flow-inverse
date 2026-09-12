"""Synthetic image pairs with exact ground-truth optical flow.

Conventions used throughout the package
---------------------------------------
* Images are 2-D float arrays indexed ``img[y, x]``.
* A flow field is a pair ``(u, v)`` of arrays the same shape as the image,
  where ``u`` is the x-displacement and ``v`` the y-displacement, both in
  pixels, defined at frame-0 pixel positions.
* Brightness constancy: ``I1(x + u, y + v) = I0(x, y)``.

To build a pair with *exact* flow we choose motions whose inverse map is
analytic: frame 1 is produced by sampling frame 0 at ``finv(q)`` for every
frame-1 pixel ``q``, and the flow at frame-0 pixel ``p`` is ``f(p) - p``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def make_texture(shape: tuple[int, int], sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """Smooth random texture in [0, 1].

    Gaussian-filtered white noise has gradients in every direction, so the
    aperture problem is mild and the flow is recoverable almost everywhere.
    ``sigma`` controls the feature scale: larger values give a smoother image
    that is easier to match but carries less information.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    tex = gaussian_filter(noise, sigma)
    tex -= tex.min()
    tex /= tex.max()
    return tex


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    return xs.astype(float), ys.astype(float)


def warp(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, order: int = 3) -> np.ndarray:
    """Sample ``img`` at continuous coordinates ``(xs, ys)`` with spline interpolation."""
    return map_coordinates(img, [ys, xs], order=order, mode="reflect")


def translation_flow(shape: tuple[int, int], dx: float, dy: float):
    """Uniform translation by ``(dx, dy)`` pixels."""
    u = np.full(shape, float(dx))
    v = np.full(shape, float(dy))

    def finv(xq, yq):
        return xq - dx, yq - dy

    return (u, v), finv


def rotation_flow(shape: tuple[int, int], angle_deg: float, center=None):
    """Rigid rotation by ``angle_deg`` about ``center`` (default: image centre)."""
    xs, ys = _grid(shape)
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    dxp, dyp = xs - cx, ys - cy
    u = (c * dxp - s * dyp) - dxp
    v = (s * dxp + c * dyp) - dyp

    def finv(xq, yq):
        dxq, dyq = xq - cx, yq - cy
        return cx + c * dxq + s * dyq, cy - s * dxq + c * dyq

    return (u, v), finv


def scaling_flow(shape: tuple[int, int], scale: float, center=None):
    """Isotropic expansion (``scale > 1``) or contraction about ``center``."""
    xs, ys = _grid(shape)
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    u = (scale - 1.0) * (xs - cx)
    v = (scale - 1.0) * (ys - cy)

    def finv(xq, yq):
        return cx + (xq - cx) / scale, cy + (yq - cy) / scale

    return (u, v), finv


def make_pair(texture: np.ndarray, finv, noise_std: float = 0.0, seed: int = 1):
    """Frame pair ``(I0, I1)`` where ``I1(q) = I0(finv(q))``, plus optional noise."""
    xs, ys = _grid(texture.shape)
    x0, y0 = finv(xs, ys)
    i0 = texture.copy()
    i1 = warp(texture, x0, y0)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        i0 = i0 + rng.normal(0, noise_std, i0.shape)
        i1 = i1 + rng.normal(0, noise_std, i1.shape)
    return i0, i1


def make_disc(shape: tuple[int, int], center=None, radius: float = 20.0, ring: float = 3.0) -> np.ndarray:
    """Piecewise-constant disc with a grey ring, as in the dissertation's phantoms.

    Piecewise-constant images have zero gradient almost everywhere, so the
    flow cannot be recovered or applied without adding texture (§3.3); this
    phantom exists to demonstrate that.
    """
    xs, ys = _grid(shape)
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    r = np.hypot(xs - cx, ys - cy)
    img = np.zeros(shape)
    img[r <= radius + ring] = 0.5
    img[r <= radius] = 1.0
    return img


def make_phantom(shape: tuple[int, int], n_blobs: int = 7, texture: float = 0.08, seed: int = 0) -> np.ndarray:
    """Soft-edged discs and Gaussian blobs of varied size and brightness on a textured background.

    Has visible structure (so motion is easy to see) and gradients everywhere
    (so the flow is recoverable) -- the synthetic stand-in for an MRI slice.
    Discs have a ~2 px edge so the image stays differentiable.
    """
    rng = np.random.default_rng(seed)
    xs, ys = _grid(shape)
    img = texture * make_texture(shape, sigma=2.0, seed=seed + 1)
    for k in range(n_blobs):
        cx = rng.uniform(0.2, 0.8) * shape[1]
        cy = rng.uniform(0.2, 0.8) * shape[0]
        r = rng.uniform(0.04, 0.11) * shape[1]
        amp = rng.uniform(0.35, 1.0)
        if k % 2 == 0:  # soft-edged disc, brighter interior texture
            d = np.hypot(xs - cx, ys - cy)
            img += amp * 0.5 * (1 - np.tanh((d - r) / 1.5))
        else:  # elliptical Gaussian blob
            sy = r * rng.uniform(0.6, 1.6)
            img += amp * np.exp(-0.5 * (((xs - cx) / r) ** 2 + ((ys - cy) / sy) ** 2))
    img -= img.min()
    img /= img.max()
    return img


def rigid_flow(shape: tuple[int, int], angle_deg: float, dx: float, dy: float, center=None):
    """Rotation about ``center`` followed by translation: ``p -> c + R(p - c) + d``."""
    (u, v), _ = rotation_flow(shape, angle_deg, center)
    u, v = u + dx, v + dy
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)

    def finv(xq, yq):
        dxq, dyq = xq - dx - cx, yq - dy - cy
        return cx + c * dxq + s * dyq, cy - s * dxq + c * dyq

    return (u, v), finv
