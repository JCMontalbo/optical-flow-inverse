"""Generating new images from a recovered flow field.

This is the part of the dissertation that goes beyond estimating motion: once
``(u, v)`` is known, it is *data*. Transforming it in principled ways and
propagating frame 0 along the result yields new images that respect the
geometry of the scene -- variation only where motion has been shown to be
possible, unlike blind rotate/skew/shear augmentation.

Section references are to J. Montalbo, *Inverse Problems and Forward
Propagation of Optical Flow*, UT Arlington, 2020.

* §3.1  ``scale_flow``            -- global homotopy (u, v) -> (εu, εv)
* §3.2  ``gaussian_window``,
        ``localized_family``      -- change only inside a Gaussian window
* §3.3  ``advection_norms``       -- how well a pair satisfies brightness constancy
* §4.2  ``propagate_ode``         -- explicit pixel movement as an ODE, with
                                     standard and non-standard Euler steps
* §4.4  ``hybrid_evolve``         -- re-estimate, smooth, move, repeat
* §4.5  ``linear_flow``,
        ``area_preserving_perturbation``
                                  -- special-affine (area-preserving) motions
                                     from traceless generators, and a
                                     windowed, divergence-free extension
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter

from .horn_schunck import derivatives, horn_schunck, horn_schunck_pyramid
from .propagate import propagate_semi_lagrangian
from .synthetic import warp

# ----------------------------------------------------------------------------
# §3.1  Global transformation of (u, v)
# ----------------------------------------------------------------------------


def scale_flow(u: np.ndarray, v: np.ndarray, eps: float):
    """``(u, v) -> (εu, εv)``: a homotopy from frame 0 (ε = 0) to frame 1 (ε = 1).

    Under the *linearised* reconstruction ``Ẽ₂ = E₁ - Δt (Eₓ εu + E_y εv)``
    this collapses to the linear blend ``εE₂ + (1 - ε)E₁`` (§3.1) -- regions
    dim and brighten rather than move. Propagating along ``(εu, εv)`` with
    :func:`ofi.propagate_semi_lagrangian` instead transports intensity, so the
    same family genuinely moves. ``linearized_reconstruction`` is provided to
    make that difference visible.
    """
    return eps * u, eps * v


def linearized_reconstruction(i0: np.ndarray, i1: np.ndarray, u: np.ndarray, v: np.ndarray, sigma: float = 1.0):
    """``Ẽ₂ = E₁ - (Eₓ u + E_y v)`` (eq. 2.30, Δt = 1): first-order transport of frame 0."""
    ix, iy, _ = derivatives(i0, i1, sigma)
    return i0 - (ix * u + iy * v)


# ----------------------------------------------------------------------------
# §3.2  Localisation of change
# ----------------------------------------------------------------------------


def gaussian_window(shape, center, width: float) -> np.ndarray:
    """``f(x, y) = exp(-((x-h)² + (y-k)²) / (2 width²))`` centred at ``center = (h, k)``.

    The dissertation writes the exponent as ``-σ² r²`` (eq. 3.2); ``width`` here
    is the standard deviation in pixels, ``width = 1 / (σ √2)``.
    """
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    h, k = center
    return np.exp(-((xs - h) ** 2 + (ys - k) ** 2) / (2.0 * width**2))


def localized_family(u: np.ndarray, v: np.ndarray, n: int, width: float, rng=None, gain: float = 1.0):
    """One flow field per randomly placed window: ``(f_s u, f_s v)`` for ``s = 1..n``.

    Each member moves only the region under its window, so a pair of frames
    yields ``n`` distinct new images with the same underlying motion family.
    Returns a list of ``(center, u_s, v_s)``.
    """
    rng = np.random.default_rng(rng)
    out = []
    for _ in range(n):
        h = rng.uniform(0, u.shape[1] - 1)
        k = rng.uniform(0, u.shape[0] - 1)
        f = gain * gaussian_window(u.shape, (h, k), width)
        out.append(((h, k), f * u, f * v))
    return out


# ----------------------------------------------------------------------------
# §3.3  Advection-term diagnostics
# ----------------------------------------------------------------------------


def advection_norms(i0: np.ndarray, i1: np.ndarray, u: np.ndarray, v: np.ndarray, sigma: float = 1.0) -> dict:
    """The five norms of Table 3.1: how faithfully a pair satisfies brightness constancy.

    ``NormAdvec = ‖Eₓu + E_y v + E_t‖`` is the residual of the constraint;
    ``NormRel`` is the relative error of the linearised reconstruction.

    The dissertation's point (§3.3): for piecewise-constant images
    ``Eₓ = E_y = 0`` almost everywhere, so the flow has nothing to act on and
    the constraint carries no information there; adding texture ("noise")
    makes the motion observable. The exact numbers in Tables 3.1–3.5 depend
    on the derivative stencil (one-sided from frame 0 there; centred on the
    average frame here), so this function reproduces the diagnostic, not the
    table. The observability claim itself is tested directly in
    ``tests/test_augment.py`` via the recovered flow inside a rotating disc.
    """
    ix, iy, it = derivatives(i0, i1, sigma)
    rec = i0 - (ix * u + iy * v)
    return {
        "NormExEy": float(np.sqrt((ix**2).sum() + (iy**2).sum())),
        "NormEt": float(np.linalg.norm(it)),
        "NormAdvec": float(np.linalg.norm(ix * u + iy * v + it)),
        "NormDiff": float(np.linalg.norm(rec - i1)),
        "NormRel": float(np.linalg.norm(rec - i1) / np.linalg.norm(i1)),
    }


# ----------------------------------------------------------------------------
# §4.2  Explicit pixel movement as an ODE
# ----------------------------------------------------------------------------


def propagate_ode(
    i0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    t: float,
    steps: int = 16,
    method: str = "euler",
    gamma: float = 0.5,
) -> np.ndarray:
    """Evolve frame 0 along ``dx/dt = u(x, y), dy/dt = v(x, y)`` to time ``t``.

    §4.2 moves each pixel forward with Euler steps ``x ← x + Δt·u``. Forward
    (scatter) movement leaves holes and speckles (§4.3); tracing the same
    characteristics *backward* from every output pixel is the well-posed
    twin and gives identical motion, so that is what is done here.

    ``method="nonstandard"`` uses the non-standard Euler step of §4.4,
    ``x ← x + (1 - e^{-γΔt}) u / γ`` (after Dimitrov & Kojouharov), which
    damps large displacements and behaves differently late in long runs.
    ``method="midpoint"`` is second-order Runge-Kutta for when accuracy
    matters more than fidelity to the dissertation's scheme.

    Note the two readings of a field: :func:`ofi.propagate_semi_lagrangian`
    treats ``(u, v)`` as a *displacement* (exact for the synthetic pairs and
    for the Horn-Schunck output); this function treats it as a *velocity*
    (exact for the §4.5 generators). They agree to first order in ``|u, v|``.
    """
    ys, xs = np.mgrid[0 : i0.shape[0], 0 : i0.shape[1]].astype(float)
    px, py = xs.copy(), ys.copy()
    dt = t / steps
    if method == "euler":
        h = dt
    elif method == "nonstandard":
        h = (1.0 - np.exp(-gamma * dt)) / gamma
    elif method == "midpoint":
        h = dt
    else:
        raise ValueError(method)

    def vel(x, y):
        return warp(u, x, y, order=1), warp(v, x, y, order=1)

    for _ in range(steps):
        ux, vy = vel(px, py)
        if method == "midpoint":
            ux, vy = vel(px - 0.5 * h * ux, py - 0.5 * h * vy)
        px = px - h * ux
        py = py - h * vy
    return warp(i0, px, py, order=3)


# ----------------------------------------------------------------------------
# §4.4  Forward hybrid propagation
# ----------------------------------------------------------------------------


@dataclass
class HybridResult:
    frames: list  # evolved images, frames[0] is i0
    rel_error: np.ndarray  # ‖frame_k - i1‖ / ‖i1‖ per step (Fig. 4.13 / 4.17)


def hybrid_evolve(
    i0: np.ndarray,
    i1: np.ndarray,
    n_steps: int = 10,
    step: float = 0.5,
    alpha: float = 0.1,
    smooth: float = 2.0,
    hs_iters: int = 300,
    levels: int = 1,
) -> HybridResult:
    """Deform frame 0 into frame 1 by repeatedly re-estimating and following the flow.

    The single flow between the two frames is inaccurate for large or
    non-uniform motion, so §4.2–4.4 recompute ``(u, v)`` between the *current*
    evolved image and the target after every movement step, Gaussian-smooth
    the field so pixels move coherently, and take a short step along it. The
    relative error to the target decreases monotonically (Fig. 4.17); the
    intermediate frames are the "in-between" data.

    The dissertation uses single-scale Horn-Schunck inside the loop
    (``levels=1``); the loop is then a *temporal* analogue of coarse-to-fine,
    keeping each linearisation valid by taking short steps. ``levels > 1``
    uses the spatial pyramid inside the loop as well.
    """
    cur = i0.astype(float).copy()
    frames = [cur]
    errs = [float(np.linalg.norm(cur - i1) / np.linalg.norm(i1))]
    for _ in range(n_steps):
        if levels > 1:
            u, v = horn_schunck_pyramid(cur, i1, alpha=alpha, levels=levels, n_iter=hs_iters)
        else:
            u, v, _ = horn_schunck(cur, i1, alpha=alpha, n_iter=hs_iters)
        if smooth > 0:
            u = gaussian_filter(u, smooth)
            v = gaussian_filter(v, smooth)
        cur = propagate_semi_lagrangian(cur, u, v, step)
        frames.append(cur)
        errs.append(float(np.linalg.norm(cur - i1) / np.linalg.norm(i1)))
    return HybridResult(frames, np.asarray(errs))


# ----------------------------------------------------------------------------
# §4.5  Area-preserving transformations
# ----------------------------------------------------------------------------

# Generators of the special affine group (eq. 4.15–4.18). All have trace 0.
ROTATION = np.array([[0.0, -1.0], [1.0, 0.0]])
SQUEEZE = np.array([[1.0, 0.0], [0.0, -1.0]])
SHEAR = np.array([[0.0, 1.0], [0.0, 0.0]])
SADDLE = np.array([[1.0, 1.0], [3.0, -1.0]])  # Example 2: λ = ±2


def linear_flow(shape, A: np.ndarray, t: float, center=None):
    """Displacement field of the linear system ``x' = A (x - c)`` over time ``t``.

    The solution is ``x(t) = c + exp(At)(x₀ - c)`` (eq. 4.11–4.13), and
    ``det exp(At) = exp(t·tr A)``, so the map preserves area exactly when
    ``tr A = 0``, i.e. ``λ₁ = -λ₂`` (eq. 4.14). Returns ``(u, v)`` such that
    ``x₀ + (u, v)`` is where the pixel at ``x₀`` ends up.
    """
    A = np.asarray(A, dtype=float)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    B = expm(A * t)
    dx, dy = xs - cx, ys - cy
    u = (B[0, 0] - 1) * dx + B[0, 1] * dy
    v = B[1, 0] * dx + (B[1, 1] - 1) * dy
    return u, v


def stream_function(shape, A: np.ndarray, center=None) -> np.ndarray:
    """ψ with ``∇⊥ψ = (ψ_y, -ψ_x) = A (x - c)`` -- exists iff ``tr A = 0``.

    For ``A = [[a, b], [c, -a]]``: ``ψ = a·xy + b·y²/2 - c·x²/2``.
    """
    A = np.asarray(A, dtype=float)
    if abs(np.trace(A)) > 1e-12:
        raise ValueError("a stream function exists only for traceless (divergence-free) A")
    a, b, c = A[0, 0], A[0, 1], A[1, 0]
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    cx, cy = center if center is not None else ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
    dx, dy = xs - cx, ys - cy
    return a * dx * dy + 0.5 * b * dy**2 - 0.5 * c * dx**2


def area_preserving_perturbation(shape, A: np.ndarray, center, width: float, amplitude: float = 1.0):
    """A *localised* area-preserving velocity field: ``∇⊥(f · ψ_A)``.

    Extension of §4.5. Windowing the linear field directly (``f · A x``, as in
    §3.2) breaks divergence-freeness because ``∇f · Ax ≠ 0``; windowing the
    stream function and then taking the perpendicular gradient keeps the
    field exactly divergence-free, so the motion it generates is still
    area-preserving while acting only near ``center``. ``amplitude`` is in
    pixels of displacement per unit time at the window centre-scale.
    """
    psi = gaussian_window(shape, center, width) * stream_function(shape, A, center)
    psi_y, psi_x = np.gradient(psi)
    u, v = psi_y, -psi_x
    peak = max(float(np.hypot(u, v).max()), 1e-12)
    return amplitude * u / peak, amplitude * v / peak


def jacobian_determinant(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``det(I + ∇(u, v))`` per pixel: local area change of the map ``x ↦ x + (u, v)``."""
    uy, ux = np.gradient(u)
    vy, vx = np.gradient(v)
    return (1 + ux) * (1 + vy) - uy * vx


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``∂u/∂x + ∂v/∂y`` -- zero for an area-preserving velocity field."""
    return np.gradient(u, axis=1) + np.gradient(v, axis=0)
