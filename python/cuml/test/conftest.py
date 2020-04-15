import pytest
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import cupy as cp
from cuml.test.golden_values import Golden, GoldenModule

def pytest_configure(config):
    cp.cuda.set_allocator(None)


@pytest.fixture(scope="module")
def nlp_20news():
    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


def pytest_addoption(parser):
    parser.addoption("--run_stress", action="store_true",
                     default=False, help="run stress tests")

    parser.addoption("--run_quality", action="store_true",
                     default=False, help="run quality tests")

    parser.addoption("--run_unit", action="store_true",
                     default=False, help="run unit tests")

    parser.addoption(
        "--golden-values", action="store", default="read",
        choices=["read", "check", "recompute"],
        help="Golden value behavior: read, check, recompute"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_quality"):
        # --run_quality given in cli: do not skip quality tests
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)
        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        return

    else:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if config.getoption("--run_stress"):
        # --run_stress given in cli: do not skip stress tests

        skip_unit = pytest.mark.skip(
            reason="Stress tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

        return

    else:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)


@pytest.fixture(scope="module")
def golden_option(request):
    return request.config.getoption("--golden-values")

@pytest.fixture(scope="module")
def golden_module(request, golden_option):
    gmodule = GoldenModule(request.node.name,
                           os.path.dirname(request.node.fspath),
                           golden_option)
    yield gmodule
    gmodule.cleanup()

@pytest.fixture()
def golden(request, golden_module):
    """Allow golden (precomputed) values for PyTests

    Golden values can be precomputed by one run, saved in JSON files, and
    reused for comparisons in future runs.

    Tests with long-running code should check on each run whether or not
    to rerun the slow code by checking `golden.recompute`.

    Attributes can be stored (when recomputing) on this golden object by
    setting attributes that end with an underscore, like:
    `golden.accuracy_ = 100`. Currently only JSON-seralizable types
    are supported, so numpy arrays must be converted to lists before storing.

    Running modes:
      You can specify the golden value running mode as a command-line flag:
         pytest --golden-values=[mymode]

      Available modes are:
        * "read" (default): Reuse cached golden values do not recompute
        * "recompute": Recompute golden values and store to files
        * "check": Recompute golden values and check that they have not changed.
                   Currently uses a strict equality check. Does NOT save.

    Examples
    --------
    .. code-block:: python

        def test_something(golden):
          if golden.recompute:
            golden.my_value_ = run_long_operation()

          assert golden.my_value_ == comparison_fast_operation()

    """
    return Golden(golden_module, request.node.name)
