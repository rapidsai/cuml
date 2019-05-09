
import pytest


def pytest_addoption(parser):
    parser.addoption("--run_stress", action="store_true",
                     default=False, help="run stress tests")
    
    parser.addoption("--run_quality", action="store_true",
                     default=False, help="run quality tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests
        return

    skip_stress = pytest.mark.skip(reason="Stress tests run with --run_stress flag." )
    for item in items:
        if "stress" in item.keywords:
            item.add_marker(skip_stress)
    
    if config.getoption("--run_quality"):
        # --run_quality given in cli: do not skip quality tests
        return

    skip_quality = pytest.mark.skip(reason="Quality tests run with --run_quality flag." )
    for item in items:
        if "quality" in item.keywords:
            item.add_marker(skip_quality)
