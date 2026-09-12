"""Carrying labels along a flow field.

A segmentation label is just another image defined on the same grid, so it
can be transported along a flow exactly as the intensities are. That is what
makes flow-based data generation useful for training: every synthetic frame
comes with its label for free, and a label drawn on one video frame can be
propagated to its neighbours.

Hard labels are transported as one-hot channels with linear interpolation and
re-hardened by argmax, which keeps class boundaries sharp and never invents a
class that was not present.
"""

from __future__ import annotations

import numpy as np

from .propagate import propagate_semi_lagrangian


def propagate_label(label: np.ndarray, u: np.ndarray, v: np.ndarray, t: float = 1.0, n_classes: int | None = None) -> np.ndarray:
    """Transport an integer (or boolean) label map along ``(u, v)`` to time ``t``."""
    if label.dtype == bool:
        soft = _propagate_soft(label.astype(float), u, v, t)
        return soft > 0.5
    label = label.astype(int)
    n = int(n_classes if n_classes is not None else label.max() + 1)
    stack = np.stack([_propagate_soft((label == c).astype(float), u, v, t) for c in range(n)])
    return stack.argmax(axis=0)


def _propagate_soft(chan: np.ndarray, u, v, t) -> np.ndarray:
    # linear interpolation: soft memberships stay in [0, 1] and sum to 1 across classes
    return propagate_semi_lagrangian(chan, u, v, t, order=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0



def chain_label(label: np.ndarray, flows, t: float = 1.0) -> np.ndarray:
    """Propagate a boolean or integer label through a sequence of flows ``[(u, v), ...]``.

    Memberships are kept *soft* between steps and hardened only at the end,
    so thin structures are not eroded by re-thresholding at every frame.
    """
    if label.dtype == bool:
        soft = label.astype(float)
        for u, v in flows:
            soft = _propagate_soft(soft, u, v, t)
        return soft > 0.5
    label = label.astype(int)
    n = int(label.max() + 1)
    stack = np.stack([(label == c).astype(float) for c in range(n)])
    for u, v in flows:
        stack = np.stack([_propagate_soft(ch, u, v, t) for ch in stack])
    return stack.argmax(axis=0)
