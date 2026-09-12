"""Error measures for flow fields and images."""

from __future__ import annotations

import numpy as np


def _interior(a: np.ndarray, border: int) -> np.ndarray:
    return a[border:-border, border:-border] if border > 0 else a


def endpoint_error(u, v, u_true, v_true, border: int = 8) -> float:
    """Mean Euclidean distance between estimated and true displacement vectors.

    ``border`` pixels are excluded on each side: the image boundary has no
    incoming information, so every method degrades there and the number would
    otherwise measure boundary handling rather than the solver.
    """
    e = np.hypot(u - u_true, v - v_true)
    return float(_interior(e, border).mean())


def angular_error(u, v, u_true, v_true, border: int = 8) -> float:
    """Mean angular error (degrees) of the 3-vector (u, v, 1), as in Barron et al."""
    num = u * u_true + v * v_true + 1.0
    den = np.sqrt(u**2 + v**2 + 1.0) * np.sqrt(u_true**2 + v_true**2 + 1.0)
    ang = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))
    return float(_interior(ang, border).mean())


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0, border: int = 8) -> float:
    """Peak signal-to-noise ratio in dB between two images on the same scale."""
    mse = float((_interior(a - b, border) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak**2 / mse)
