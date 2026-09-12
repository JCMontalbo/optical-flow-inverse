import numpy as np

from ofi import make_pair, rotation_flow, scaling_flow, translation_flow, warp
from tests.conftest import SHAPE


def test_texture_range(texture):
    assert texture.shape == SHAPE
    assert texture.min() == 0.0 and texture.max() == 1.0


def test_pair_satisfies_brightness_constancy(texture):
    """I1 sampled at p + flow(p) must equal I0(p) for every synthetic motion."""
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(float)
    for (u, v), finv in [
        translation_flow(SHAPE, 1.3, -0.7),
        rotation_flow(SHAPE, 3.0),
        scaling_flow(SHAPE, 1.05),
    ]:
        i0, i1 = make_pair(texture, finv)
        back = warp(i1, xs + u, ys + v)
        err = np.abs(back - i0)[10:-10, 10:-10]
        assert err.max() < 1e-2


def test_noise_is_added(texture):
    _, finv = translation_flow(SHAPE, 1.0, 0.0)
    clean = make_pair(texture, finv)[0]
    noisy = make_pair(texture, finv, noise_std=0.05)[0]
    assert np.abs(noisy - clean).std() > 0.01
