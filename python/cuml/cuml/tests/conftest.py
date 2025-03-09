#
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.testing.utils import create_synthetic_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.datasets import make_regression as skl_make_reg
from sklearn.datasets import make_classification as skl_make_clas
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
from datetime import timedelta
from math import ceil
from ssl import create_default_context
from urllib.request import build_opener, HTTPSHandler, install_opener
import certifi
import functools
import hypothesis
from cuml.internals.safe_imports import gpu_only_import
import pytest
import os
import subprocess
import time
import pandas as pd
import cudf.pandas

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = "cuml.testing.plugins.quick_run_plugin"


# Install SSL certificates
def pytest_sessionstart(session):
    ssl_context = create_default_context(cafile=certifi.where())
    https_handler = HTTPSHandler(context=ssl_context)
    install_opener(build_opener(https_handler))


CI = os.environ.get("CI") in ("true", "1")
HYPOTHESIS_ENABLED = os.environ.get("HYPOTHESIS_ENABLED") in (
    "true",
    "1",
)


# Configure hypothesis profiles

HEALTH_CHECKS_SUPPRESSED_BY_DEFAULT = (
    list(hypothesis.HealthCheck)
    if CI
    else [
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.too_slow,
    ]
)


HYPOTHESIS_DEFAULT_PHASES = (
    (
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
        hypothesis.Phase.shrink,
    )
    if HYPOTHESIS_ENABLED
    else (hypothesis.Phase.explicit,)
)


hypothesis.settings.register_profile(
    name="unit",
    deadline=None if CI else timedelta(milliseconds=2000),
    parent=hypothesis.settings.get_profile("default"),
    phases=HYPOTHESIS_DEFAULT_PHASES,
    max_examples=20,
    suppress_health_check=HEALTH_CHECKS_SUPPRESSED_BY_DEFAULT,
)

hypothesis.settings.register_profile(
    name="quality",
    parent=hypothesis.settings.get_profile("unit"),
    max_examples=100,
)

hypothesis.settings.register_profile(
    name="stress",
    parent=hypothesis.settings.get_profile("quality"),
    max_examples=200,
)


def pytest_addoption(parser):
    # Any custom option, that should be available at any time (not just a
    # plugin), goes here.

    group = parser.getgroup("cuML Custom Options")

    group.addoption(
        "--run_stress",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'stress'. These are the most intense "
            "tests that take the longest to run and are designed to stress "
            "the hardware's compute resources."
        ),
    )

    group.addoption(
        "--run_quality",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'quality'. These tests are more "
            "computationally intense than 'unit', but less than 'stress'"
        ),
    )

    group.addoption(
        "--run_unit",
        action="store_true",
        default=False,
        help=(
            "Runs tests marked with 'unit'. These are the quickest tests "
            "that are focused on accuracy and correctness."
        ),
    )


def pytest_collection_modifyitems(config, items):

    # Check for hypothesis tests without examples
    tests_without_examples = []
    for item in items:
        if isinstance(item, pytest.Function):
            # Check if function has @given decorator
            has_given = hasattr(item.obj, "hypothesis")
            # Check if function has @example decorator
            has_example = hasattr(item.obj, "hypothesis_explicit_examples")

            if has_given and not has_example:
                tests_without_examples.append(
                    f"Test {item.name} uses @given but has no @example cases."
                )

    if tests_without_examples:
        msg = (
            "\nCollection failed because the following tests lack examples:\n"
            + "\n".join(f"  - {e}" for e in tests_without_examples)
        )
        raise pytest.UsageError(msg)

    # Handle test categories (unit/quality/stress)
    should_run_quality = config.getoption("--run_quality")
    should_run_stress = config.getoption("--run_stress")

    # Run unit is implied if no --run_XXX is set
    should_run_unit = config.getoption("--run_unit") or not (
        should_run_quality or should_run_stress
    )

    # Mark the tests as "skip" if needed
    if not should_run_unit:
        skip_unit = pytest.mark.skip(
            reason="Unit tests run with --run_unit flag."
        )
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    if not should_run_quality:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag."
        )
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if not should_run_stress:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag."
        )
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cudf_pandas: mark test as requiring the cudf.pandas wrapper",
    )
    cp.cuda.set_allocator(None)
    # max_gpu_memory: Capacity of the GPU memory in GB
    pytest.max_gpu_memory = get_gpu_memory()
    pytest.adapt_stress_test = "CUML_ADAPT_STRESS_TESTS" in os.environ

    # Load special hypothesis profiles for either quality or stress tests.
    # Note that the profile can be manually overwritten with the
    # --hypothesis-profile command line option in which case the settings
    # specified here will be ignored.
    if config.getoption("--run_stress"):
        hypothesis.settings.load_profile("stress")
    elif config.getoption("--run_quality"):
        hypothesis.settings.load_profile("quality")
    else:
        hypothesis.settings.load_profile("unit")


def pytest_pyfunc_call(pyfuncitem):
    """Skip tests that require the cudf.pandas accelerator

    Tests marked with `@pytest.mark.cudf_pandas` will only be run if the
    cudf.pandas accelerator is enabled via the `cudf.pandas` plugin.
    """
    if "cudf_pandas" in pyfuncitem.keywords and not cudf.pandas.LOADED:
        pytest.skip("Test requires cudf.pandas accelerator")


@pytest.fixture(scope="session")
def nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(
            subset="train", shuffle=True, random_state=42
        )
    except:  # noqa E722
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="session")
def housing_dataset():
    try:
        data = fetch_california_housing()

    # failing to download has appeared as multiple varied errors in CI
    except:  # noqa E722
        pytest.xfail(reason="Error fetching housing dataset")

    X = cp.array(data["data"])
    y = cp.array(data["target"])

    feature_names = data["feature_names"]

    return X, y, feature_names


@functools.cache
def get_boston_data():
    n_retries = 3
    url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/datasets/data/boston_house_prices.csv"  # noqa: E501
    for _ in range(n_retries):
        try:
            return pd.read_csv(url, header=None)
        except Exception:
            time.sleep(1)
    raise RuntimeError(
        f"Failed to download file from {url} after {n_retries} retries."
    )


@pytest.fixture(scope="session")
def deprecated_boston_dataset():
    # dataset was removed in Scikit-learn 1.2, we should change it for a
    # better dataset for tests, see
    # https://github.com/rapidsai/cuml/issues/5158

    try:
        df = get_boston_data()
    except:  # noqa E722
        pytest.xfail(reason="Error fetching Boston housing dataset")
    n_samples = int(df[0][0])
    data = df[list(np.arange(13))].values[2:n_samples].astype(np.float64)
    targets = df[13].values[2:n_samples].astype(np.float64)

    return Bunch(
        data=data,
        target=targets,
    )


@pytest.fixture(
    scope="session",
    params=["digits", "deprecated_boston_dataset", "diabetes", "cancer"],
)
def test_datasets(request, deprecated_boston_dataset):
    test_datasets_dict = {
        "digits": datasets.load_digits(),
        "deprecated_boston_dataset": deprecated_boston_dataset,
        "diabetes": datasets.load_diabetes(),
        "cancer": datasets.load_breast_cancer(),
    }

    return test_datasets_dict[request.param]


@pytest.fixture(scope="session")
def random_seed(request):
    current_random_seed = os.getenv("PYTEST_RANDOM_SEED")
    if current_random_seed is not None and current_random_seed.isdigit():
        random_seed = int(current_random_seed)
    else:
        random_seed = np.random.randint(0, 1e6)
        os.environ["PYTEST_RANDOM_SEED"] = str(random_seed)
    print("\nRandom seed value:", random_seed)
    return random_seed


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(scope="function")
def failure_logger(request):
    """
    To be used when willing to log the random seed used in some failing test.
    """
    yield
    if request.node.rep_call.failed:
        error_msg = " {} failed with seed: {}"
        error_msg = error_msg.format(
            request.node.nodeid, os.getenv("PYTEST_RANDOM_SEED")
        )
        print(error_msg)


@pytest.fixture(scope="session")
def exact_shap_regression_dataset():
    return create_synthetic_dataset(
        generator=skl_make_reg,
        n_samples=101,
        n_features=11,
        test_size=3,
        random_state_generator=42,
        random_state_train_test_split=42,
        noise=0.1,
    )


@pytest.fixture(scope="session")
def exact_shap_classification_dataset():
    return create_synthetic_dataset(
        generator=skl_make_clas,
        n_samples=101,
        n_features=11,
        test_size=3,
        random_state_generator=42,
        random_state_train_test_split=42,
    )


def get_gpu_memory():
    bash_command = "nvidia-smi --query-gpu=memory.total --format=csv"
    output = subprocess.check_output(bash_command, shell=True).decode("utf-8")
    lines = output.split("\n")
    lines.pop(0)
    gpus_memory = []
    for line in lines:
        tokens = line.split(" ")
        if len(tokens) > 1:
            gpus_memory.append(int(tokens[0]))
    gpus_memory.sort()
    max_gpu_memory = ceil(gpus_memory[-1] / 1024)

    return max_gpu_memory
