"""Forward propagation of image information along a motion field.

Given frame ``I0`` and a flow ``(u, v)``, brightness constancy says intensity
is transported along the motion:

    ∂I/∂τ + u ∂I/∂x + v ∂I/∂y = 0,      I(·, 0) = I0,   τ ∈ [0, 1].

Solving this forward in time yields the image at any intermediate instant --
frame interpolation without a second frame. Two discretisations are provided:

* ``propagate_semi_lagrangian`` -- trace each pixel back along the flow and
  interpolate. Unconditionally stable, non-diffusive, and exact when the flow
  is the true displacement.
* ``propagate_upwind`` -- explicit first-order upwind finite differences with
  CFL-limited sub-steps. Simple and conservative, but numerically diffusive:
  the sharper the image, the more it blurs. Included to make that trade-off
  measurable.
"""

from __future__ import annotations

import numpy as np

from .synthetic import warp


def propagate_semi_lagrangian(
    i0: np.ndarray, u: np.ndarray, v: np.ndarray, t: float, fixed_point_iters: int = 3, order: int = 3
) -> np.ndarray:
    """Image at time ``t`` in [0, 1] by backward characteristic tracing.

    A pixel at ``q`` in the output came from ``p = q - t * flow(p)`` in frame 0.
    Because the flow is defined at the *departure* point ``p``, we solve for
    ``p`` by a few fixed-point iterations rather than evaluating the flow at
    ``q``; this is the difference between a first-order and an exact inverse
    for smooth fields. ``order`` is the interpolation order of the final
    resampling (3 for images; 1 for soft label channels, which must stay in [0, 1]).
    """
    ys, xs = np.mgrid[0 : i0.shape[0], 0 : i0.shape[1]].astype(float)
    px, py = xs.copy(), ys.copy()
    for _ in range(fixed_point_iters):
        up = warp(u, px, py, order=1)
        vp = warp(v, px, py, order=1)
        px = xs - t * up
        py = ys - t * vp
    return warp(i0, px, py, order=order)


def propagate_upwind(i0: np.ndarray, u: np.ndarray, v: np.ndarray, t: float, cfl: float = 0.5) -> np.ndarray:
    """Image at time ``t`` by explicit upwind advection with a stationary velocity field."""
    img = i0.astype(float).copy()
    vmax = max(float(np.abs(u).max()), float(np.abs(v).max()), 1e-12)
    dt = cfl / vmax
    n_steps = int(np.ceil(t / dt))
    dt = t / max(n_steps, 1)

    for _ in range(n_steps):
        # One-sided differences chosen against the direction of transport.
        dx_back = img - np.roll(img, 1, axis=1)
        dx_fwd = np.roll(img, -1, axis=1) - img
        dy_back = img - np.roll(img, 1, axis=0)
        dy_fwd = np.roll(img, -1, axis=0) - img
        ix = np.where(u > 0, dx_back, dx_fwd)
        iy = np.where(v > 0, dy_back, dy_fwd)
        img = img - dt * (u * ix + v * iy)
    return img
