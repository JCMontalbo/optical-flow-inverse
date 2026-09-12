import numpy as np

from ofi import angular_error, endpoint_error, psnr


def test_identical_flows_have_zero_error():
    u = np.random.default_rng(0).standard_normal((32, 32))
    v = np.random.default_rng(1).standard_normal((32, 32))
    assert endpoint_error(u, v, u, v) == 0.0
    assert angular_error(u, v, u, v) < 1e-6


def test_endpoint_error_is_mean_distance():
    z = np.zeros((32, 32))
    assert np.isclose(endpoint_error(z + 3.0, z + 4.0, z, z), 5.0)


def test_psnr_identical_is_inf_and_noise_is_finite():
    a = np.random.default_rng(0).random((32, 32))
    assert psnr(a, a) == float("inf")
    assert 10 < psnr(a, a + 0.01) < 60
