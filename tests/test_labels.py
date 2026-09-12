import numpy as np

from ofi import horn_schunck_pyramid, make_disc, make_pair, make_texture, propagate_semi_lagrangian, rotation_flow
from ofi.augment import SHEAR, area_preserving_perturbation
from ofi.labels import chain_label, dice, iou, propagate_label
from tests.conftest import SHAPE


def _textured_disc():
    disc = make_disc(SHAPE, center=(50, 46), radius=22, ring=0)
    tex = make_texture(SHAPE, sigma=2.0, seed=5)
    return disc * (0.7 + 0.3 * tex), disc > 0.5


def test_label_follows_true_and_estimated_flow():
    img, mask = _textured_disc()
    (ut, vt), finv = rotation_flow(SHAPE, 6.0)
    i0, i1 = make_pair(img, finv)
    _, truth = make_pair(mask.astype(float), finv)
    truth = truth > 0.5
    assert iou(propagate_label(mask, ut, vt), truth) > 0.97
    u, v = horn_schunck_pyramid(i0, i1)
    assert iou(propagate_label(mask, u, v), truth) > 0.9


def test_multiclass_label_is_hard_and_keeps_classes():
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    label = (xs > 40).astype(int) + (ys > 50).astype(int)  # classes 0, 1, 2
    (ut, vt), _ = rotation_flow(SHAPE, 3.0)
    moved = propagate_label(label, ut, vt)
    assert moved.dtype.kind == "i"
    assert set(np.unique(moved)) <= {0, 1, 2}
    zero = np.zeros(SHAPE)
    assert np.array_equal(propagate_label(label, zero, zero), label)


def test_image_and_label_stay_consistent_under_a_perturbation():
    """The point of carrying labels: (image', label') is a valid training pair."""
    img, mask = _textured_disc()
    du, dv = area_preserving_perturbation(SHAPE, SHEAR, (50, 46), 14.0, amplitude=4.0)
    img2 = propagate_semi_lagrangian(img, du, dv, 1.0)
    mask2 = propagate_label(mask, du, dv)
    assert iou(mask2, img2 > 0.35) > 0.95
    assert iou(mask, img2 > 0.35) < iou(mask2, img2 > 0.35)  # the unmoved label is worse


def test_chain_label_identity_and_metrics():
    _, mask = _textured_disc()
    zero = np.zeros(SHAPE)
    assert np.array_equal(chain_label(mask, [(zero, zero)] * 5), mask)
    assert iou(mask, mask) == 1.0 and dice(mask, mask) == 1.0
    assert iou(mask, ~mask) == 0.0
