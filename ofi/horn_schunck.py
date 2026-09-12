"""Variational optical flow (Horn & Schunck, 1981) -- the inverse problem.

We look for the motion field ``(u, v)`` minimising

    E(u, v) = ∫ (I_x u + I_y v + I_t)^2  +  α^2 ( |∇u|^2 + |∇v|^2 ) dx

The first term is the linearised brightness-constancy constraint; on its own it
fixes only the component of the flow along the image gradient (the aperture
problem), so the inverse problem is ill-posed. The second term is Tikhonov
regularisation on the flow: it makes the problem well-posed and controls the
smoothness of the recovered field through α.

The Euler-Lagrange equations are a coupled pair of Poisson-type PDEs. With the
Laplacian discretised as ``Δu ≈ ū - u`` (``ū`` a local weighted mean) they
reduce to the classic Jacobi update

    u ← ū - I_x (I_x ū + I_y v̄ + I_t) / (α^2 + I_x^2 + I_y^2)
    v ← v̄ - I_y (I_x ū + I_y v̄ + I_t) / (α^2 + I_x^2 + I_y^2)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, zoom

from .synthetic import warp

# Horn-Schunck's weighted-average kernel for the local mean ū.
_AVG = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]], dtype=float) / 12.0


def derivatives(i0: np.ndarray, i1: np.ndarray, sigma: float = 1.0):
    """Spatial and temporal derivatives shared by both frames.

    Both frames are pre-smoothed by a Gaussian of width ``sigma``; the spatial
    derivatives are central differences of the *average* frame so that they are
    centred in time, matching the temporal difference ``I1 - I0``.
    """
    if sigma > 0:
        i0 = gaussian_filter(i0, sigma)
        i1 = gaussian_filter(i1, sigma)
    avg = 0.5 * (i0 + i1)
    iy, ix = np.gradient(avg)
    it = i1 - i0
    return ix, iy, it


def horn_schunck(
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: float = 0.1,
    n_iter: int = 2000,
    tol: float = 1e-5,
    sigma: float = 1.0,
    init=None,
):
    """Estimate the flow from ``i0`` to ``i1``.

    Parameters
    ----------
    alpha : regularisation weight. Larger values give smoother fields. Note
        that alpha has the units of an image gradient, so the right value
        depends on the intensity scale: 0.1 is a good default for images in
        [0, 1] (alpha^2 comparable to |∇I|^2); for 8-bit images scale it up.
    n_iter, tol : stop after ``n_iter`` Jacobi sweeps or once the mean update
        magnitude falls below ``tol`` pixels.
    sigma : Gaussian pre-smoothing applied before differentiation.
    init : optional ``(u, v)`` starting field (used by the pyramid).

    Returns
    -------
    u, v : flow components.
    history : mean update magnitude per iteration (for convergence plots).
    """
    ix, iy, it = derivatives(i0, i1, sigma)
    if init is None:
        u = np.zeros_like(i0)
        v = np.zeros_like(i0)
    else:
        u, v = (np.array(init[0], dtype=float), np.array(init[1], dtype=float))

    denom = alpha**2 + ix**2 + iy**2
    history = []
    for _ in range(n_iter):
        u_bar = convolve(u, _AVG, mode="nearest")
        v_bar = convolve(v, _AVG, mode="nearest")
        t = (ix * u_bar + iy * v_bar + it) / denom
        u_new = u_bar - ix * t
        v_new = v_bar - iy * t
        step = float(np.hypot(u_new - u, v_new - v).mean())
        history.append(step)
        u, v = u_new, v_new
        if step < tol:
            break
    return u, v, np.asarray(history)


def _downsample(img: np.ndarray) -> np.ndarray:
    return zoom(gaussian_filter(img, 1.0), 0.5, order=1)


def _upsample_flow(u: np.ndarray, v: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    fy = shape[0] / u.shape[0]
    fx = shape[1] / u.shape[1]
    return (zoom(u, (fy, fx), order=1) * fx, zoom(v, (fy, fx), order=1) * fy)


def horn_schunck_pyramid(
    i0: np.ndarray,
    i1: np.ndarray,
    alpha: float = 0.1,
    levels: int = 3,
    warps: int = 3,
    n_iter: int = 1000,
    tol: float = 1e-5,
    sigma: float = 1.0,
):
    """Coarse-to-fine Horn-Schunck with intermediate warping.

    The linearised constraint is only valid for sub-pixel motion. Solving on a
    Gaussian pyramid -- coarsest level first, then warping frame 1 by the
    current estimate before solving for a correction -- extends the method to
    displacements of several pixels.
    """
    pyr0, pyr1 = [i0], [i1]
    for _ in range(levels - 1):
        pyr0.append(_downsample(pyr0[-1]))
        pyr1.append(_downsample(pyr1[-1]))

    u = np.zeros_like(pyr0[-1])
    v = np.zeros_like(pyr0[-1])
    ys, xs = None, None
    for lvl in range(levels - 1, -1, -1):
        a, b = pyr0[lvl], pyr1[lvl]
        if u.shape != a.shape:
            u, v = _upsample_flow(u, v, a.shape)
        ys, xs = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
        for _ in range(warps):
            b_warped = warp(b, xs + u, ys + v, order=1)
            du, dv, _ = horn_schunck(a, b_warped, alpha, n_iter, tol, sigma)
            u, v = u + du, v + dv
    return u, v
