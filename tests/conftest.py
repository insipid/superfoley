import pytest

from tests.fixtures.make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture(tmp_path_factory):
    """Synthetic video + truth, generated once per test session (seed 0)."""
    return make_fixture(tmp_path_factory.mktemp("fixture"), seed=0)
