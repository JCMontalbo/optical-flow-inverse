import pytest

from ofi import make_texture

SHAPE = (96, 96)


@pytest.fixture(scope="session")
def texture():
    return make_texture(SHAPE, sigma=3.0, seed=0)
