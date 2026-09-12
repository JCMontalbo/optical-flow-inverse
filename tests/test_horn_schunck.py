import numpy as np

from ofi import endpoint_error, horn_schunck, horn_schunck_pyramid, make_pair, rotation_flow, translation_flow
from tests.conftest import SHAPE


def test_subpixel_translation_recovered(texture):
    (ut, vt), finv = translation_flow(SHAPE, 0.5, -0.3)
    i0, i1 = make_pair(texture, finv)
    u, v, history = horn_schunck(i0, i1)
    assert endpoint_error(u, v, ut, vt) < 0.05
    assert history[-1] < 1e-5  # converged, not just exhausted


def test_rotation_recovered(texture):
    (ut, vt), finv = rotation_flow(SHAPE, 2.0)
    i0, i1 = make_pair(texture, finv)
    u, v, _ = horn_schunck(i0, i1)
    assert endpoint_error(u, v, ut, vt) < 0.2


def test_pyramid_handles_large_motion(texture):
    """Single-scale HS fails past ~2 px; the pyramid should not."""
    (ut, vt), finv = translation_flow(SHAPE, 3.0, 2.0)
    i0, i1 = make_pair(texture, finv)
    u1, v1, _ = horn_schunck(i0, i1)
    u3, v3 = horn_schunck_pyramid(i0, i1, levels=3)
    e1 = endpoint_error(u1, v1, ut, vt)
    e3 = endpoint_error(u3, v3, ut, vt)
    assert e3 < 0.25
    assert e3 < e1 / 3


def test_regularisation_smooths(texture):
    (ut, vt), finv = translation_flow(SHAPE, 0.5, 0.0)
    i0, i1 = make_pair(texture, finv, noise_std=0.02)
    u_lo, _, _ = horn_schunck(i0, i1, alpha=0.01)
    u_hi, _, _ = horn_schunck(i0, i1, alpha=1.0, n_iter=5000)
    # Larger alpha -> smoother field (smaller gradient energy).
    assert np.gradient(u_hi)[0].std() < np.gradient(u_lo)[0].std()


def test_zero_motion_gives_zero_flow(texture):
    u, v, _ = horn_schunck(texture, texture)
    assert np.abs(u).max() < 1e-9 and np.abs(v).max() < 1e-9
