import pytest


def pytest_addoption(parser):
    parser.addoption("--run_correctness_test", action="store_true", default=False, help="run correctness tests")
    parser.addoption("--run_stress", action="store_true",
                     default=False, help="run stress tests")
@pytest.fixture
def run_stress(request):
    return request.config.getoption("--run_stress")


@pytest.fixture
def run_correctness_test(request):
    return request.config.getoption("--run_correctness_test")
