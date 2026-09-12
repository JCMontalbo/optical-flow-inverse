"""Optical flow as an inverse problem.

Two halves of one pipeline:

* ``horn_schunck`` -- the *inverse* problem: recover a motion field (u, v) from
  two frames by minimising a variational energy with PDE regularisation.
* ``propagate`` -- the *forward* problem: transport image information along a
  known motion field to synthesise frames at intermediate times.

``synthetic`` builds test pairs with exact ground-truth flow so both halves can
be measured, and ``metrics`` provides the standard error measures.
"""

from .horn_schunck import horn_schunck, horn_schunck_pyramid
from .metrics import angular_error, endpoint_error, psnr
from .propagate import propagate_semi_lagrangian, propagate_upwind
from .synthetic import (
    make_pair,
    make_texture,
    rotation_flow,
    scaling_flow,
    translation_flow,
    warp,
)

__all__ = [
    "horn_schunck",
    "horn_schunck_pyramid",
    "angular_error",
    "endpoint_error",
    "psnr",
    "propagate_semi_lagrangian",
    "propagate_upwind",
    "make_pair",
    "make_texture",
    "rotation_flow",
    "scaling_flow",
    "translation_flow",
    "warp",
]
