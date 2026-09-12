import numpy as np

from ofi import (
    horn_schunck_pyramid,
    make_disc,
    make_pair,
    make_phantom,
    make_texture,
    propagate_semi_lagrangian,
    psnr,
    rotation_flow,
)
from ofi.augment import (
    ROTATION,
    SADDLE,
    SHEAR,
    SQUEEZE,
    advection_norms,
    area_preserving_perturbation,
    divergence,
    gaussian_window,
    hybrid_evolve,
    jacobian_determinant,
    linear_flow,
    linearized_reconstruction,
    localized_family,
    propagate_ode,
    scale_flow,
    stream_function,
)
from ofi.horn_schunck import derivatives
from tests.conftest import SHAPE

GENERATORS = [ROTATION, SQUEEZE, SHEAR, SADDLE]


# 3.1 ------------------------------------------------------------------------


def test_linearized_homotopy_is_a_blend():
    """Dissertation 3.1: under the linearised model, scaling the flow by eps blends the frames."""
    ph = make_phantom(SHAPE)
    _, finv = rotation_flow(SHAPE, 2.0)
    i0, i1 = make_pair(ph, finv)
    # Any (u, v) satisfying ix*u + iy*v = -it makes the identity algebraic; use the normal flow.
    ix, iy, it = derivatives(i0, i1, sigma=1.0)
    g2 = ix**2 + iy**2 + 1e-12
    u_exact, v_exact = -it * ix / g2, -it * iy / g2
    eps = 0.3
    rec = linearized_reconstruction(i0, i1, *scale_flow(u_exact, v_exact, eps), sigma=1.0)
    assert np.allclose(rec, i0 + eps * it, atol=1e-9)


def test_transport_homotopy_moves_instead_of_blending():
    ph = make_phantom(SHAPE)
    _, finv = rotation_flow(SHAPE, 5.0)
    i0, i1 = make_pair(ph, finv)
    half = make_pair(ph, rotation_flow(SHAPE, 2.5)[1])[1]
    u, v = horn_schunck_pyramid(i0, i1)
    moved = propagate_semi_lagrangian(i0, *scale_flow(u, v, 0.5), 1.0)
    blend = 0.5 * (i0 + i1)
    assert psnr(moved, half) > psnr(blend, half) + 8


# 3.2 ------------------------------------------------------------------------


def test_gaussian_window_shape_and_peak():
    f = gaussian_window(SHAPE, (30, 40), 5.0)
    assert f.shape == SHAPE
    assert np.isclose(f[40, 30], 1.0)
    assert f[40, 40] < np.exp(-2) + 1e-9


def test_localized_family_only_moves_under_window():
    ph = make_phantom(SHAPE)
    (ut, vt), finv = rotation_flow(SHAPE, 4.0)
    i0, _ = make_pair(ph, finv)
    members = localized_family(ut, vt, n=3, width=10.0, rng=0)
    assert len(members) == 3
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    for (h, k), u_s, v_s in members:
        img = propagate_semi_lagrangian(i0, u_s, v_s, 1.0)
        far = np.hypot(xs - h, ys - k) > 40
        near = np.hypot(xs - h, ys - k) < 8
        assert np.abs(img - i0)[far].max() < 1e-3
        assert np.hypot(u_s, v_s)[near].max() > 0.5 * np.hypot(ut, vt)[near].max()


# 3.3 ------------------------------------------------------------------------


def test_flat_disc_rotation_is_unobservable_texture_makes_it_visible():
    """The observability claim behind 3.3, tested on the recovered flow."""
    (ut, vt), finv = rotation_flow(SHAPE, 5.0)
    disc = make_disc(SHAPE, radius=25, ring=0)
    tex = make_texture(SHAPE, sigma=2.0, seed=3)
    cx, cy = (SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    inside = np.hypot(xs - cx, ys - cy) < 20
    errs = {}
    for name, img in [("flat", disc), ("textured", disc * (0.7 + 0.3 * tex))]:
        a, b = make_pair(img, finv)
        u, v = horn_schunck_pyramid(a, b)
        errs[name] = np.hypot(u - ut, v - vt)[inside].mean()
    motion = np.hypot(ut, vt)[inside].mean()
    assert errs["flat"] > 0.7 * motion  # essentially nothing recovered
    assert errs["textured"] < 0.15 * motion


def test_advection_norms_keys_and_zero_motion():
    ph = make_phantom(SHAPE)
    z = np.zeros(SHAPE)
    n = advection_norms(ph, ph, z, z)
    assert set(n) == {"NormExEy", "NormEt", "NormAdvec", "NormDiff", "NormRel"}
    assert n["NormEt"] == 0.0 and n["NormAdvec"] == 0.0 and n["NormRel"] == 0.0


# 4.2 ------------------------------------------------------------------------


def test_ode_propagation_of_rotation_velocity_field():
    ph = make_phantom(SHAPE)
    th = np.deg2rad(5.0)
    _, finv = rotation_flow(SHAPE, 5.0)
    i0, i1 = make_pair(ph, finv)
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]].astype(float)
    cx, cy = (SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2
    vel_u, vel_v = -th * (ys - cy), th * (xs - cx)  # velocity field of a rotation
    euler = psnr(propagate_ode(i0, vel_u, vel_v, 1.0, steps=16, method="euler"), i1)
    mid = psnr(propagate_ode(i0, vel_u, vel_v, 1.0, steps=4, method="midpoint"), i1)
    assert euler > 60
    assert mid > euler  # second order beats first order even with fewer steps
    ns = propagate_ode(i0, vel_u, vel_v, 1.0, steps=16, method="nonstandard", gamma=0.5)
    assert psnr(ns, i1) > 30  # damped, so less accurate, but lands nearby


# 4.4 ------------------------------------------------------------------------


def test_hybrid_evolution_error_decreases():
    ph = make_phantom(SHAPE)
    _, finv = rotation_flow(SHAPE, 6.0)
    i0, i1 = make_pair(ph, finv)
    res = hybrid_evolve(i0, i1, n_steps=6, step=0.5)
    assert len(res.frames) == 7
    assert np.all(np.diff(res.rel_error) <= 1e-6)  # monotone non-increasing
    assert res.rel_error[-1] < 0.3 * res.rel_error[0]


# 4.5 ------------------------------------------------------------------------


def test_linear_flow_matches_analytic_rotation():
    th = np.deg2rad(3.0)
    (ut, vt), _ = rotation_flow(SHAPE, 3.0)
    u, v = linear_flow(SHAPE, ROTATION * th, 1.0)
    assert np.allclose(u, ut) and np.allclose(v, vt)


def test_traceless_generators_preserve_area():
    for A in GENERATORS:
        u, v = linear_flow(SHAPE, 0.05 * A, 1.0)
        J = jacobian_determinant(u, v)[4:-4, 4:-4]
        assert np.allclose(J, 1.0, atol=1e-10)
    u, v = linear_flow(SHAPE, 0.05 * np.eye(2), 1.0)  # trace 0.1: expands
    assert jacobian_determinant(u, v)[4:-4, 4:-4].mean() > 1.05


def test_stream_function_reproduces_linear_field():
    for A in GENERATORS:
        psi = stream_function(SHAPE, A)
        psi_y, psi_x = np.gradient(psi)
        u, v = linear_flow(SHAPE, A, 1e-6)  # ~ velocity * 1e-6
        assert np.allclose(psi_y[2:-2, 2:-2], u[2:-2, 2:-2] / 1e-6, atol=1e-3)
        assert np.allclose(-psi_x[2:-2, 2:-2], v[2:-2, 2:-2] / 1e-6, atol=1e-3)


def test_windowed_perturbation_is_divergence_free_and_local():
    ys, xs = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    for A in GENERATORS:
        u, v = area_preserving_perturbation(SHAPE, A, (40, 50), 8.0, amplitude=2.0)
        assert np.abs(divergence(u, v))[2:-2, 2:-2].max() < 1e-9
        assert np.isclose(np.hypot(u, v).max(), 2.0)
        far = np.hypot(xs - 40, ys - 50) > 45
        assert np.hypot(u, v)[far].max() < 1e-3  # Gaussian tail times r^2 growth of psi


def test_windowed_perturbation_preserves_disc_area():
    mask = make_disc(SHAPE, center=(48, 48), radius=14, ring=0)
    for A in GENERATORS:
        u, v = area_preserving_perturbation(SHAPE, A, (48, 48), 12.0, amplitude=3.0)
        moved = propagate_ode(mask, u, v, 1.0, steps=16, method="midpoint")
        assert abs(moved.sum() / mask.sum() - 1) < 0.01
