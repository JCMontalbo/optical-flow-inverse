import numpy as np

from ofi import make_pair, propagate_semi_lagrangian, propagate_upwind, psnr, rotation_flow, translation_flow
from tests.conftest import SHAPE


def test_t0_is_identity(texture):
    (u, v), _ = translation_flow(SHAPE, 2.0, 1.0)
    assert np.allclose(propagate_semi_lagrangian(texture, u, v, 0.0), texture)
    assert np.allclose(propagate_upwind(texture, u, v, 0.0), texture)


def test_semi_lagrangian_reaches_frame1(texture):
    for (u, v), finv in [translation_flow(SHAPE, 2.0, -1.5), rotation_flow(SHAPE, 3.0)]:
        i0, i1 = make_pair(texture, finv)
        assert psnr(propagate_semi_lagrangian(i0, u, v, 1.0), i1) > 45


def test_upwind_reaches_frame1_with_diffusion(texture):
    (u, v), finv = translation_flow(SHAPE, 2.0, -1.5)
    i0, i1 = make_pair(texture, finv)
    p_up = psnr(propagate_upwind(i0, u, v, 1.0), i1)
    p_sl = psnr(propagate_semi_lagrangian(i0, u, v, 1.0), i1)
    assert p_up > 30          # right place ...
    assert p_up < p_sl        # ... but blurrier


def test_midpoint_beats_linear_blend(texture):
    """The whole point of forward propagation: a better in-between frame."""
    (u, v), finv = translation_flow(SHAPE, 3.0, 0.0)
    i0, i1 = make_pair(texture, finv)
    _, finv_half = translation_flow(SHAPE, 1.5, 0.0)
    _, i_half = make_pair(texture, finv_half)
    assert psnr(propagate_semi_lagrangian(i0, u, v, 0.5), i_half) > psnr(0.5 * (i0 + i1), i_half) + 10
