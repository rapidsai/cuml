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

import os
import subprocess
from datetime import timedelta
from math import ceil
from ssl import create_default_context
from urllib.request import HTTPSHandler, build_opener, install_opener

import certifi
import cudf.pandas
import hypothesis
import pandas as pd
import pytest
from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
from cuml.testing.datasets import with_dtype
from sklearn import datasets
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_california_housing,
    make_classification,
    make_regression,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from tenacity import retry, stop_after_attempt, wait_exponential

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")

# =============================================================================
# Pytest Configuration
# =============================================================================

# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = "cuml.testing.plugins.quick_run_plugin"


def pytest_sessionstart(session):
    """Initialize SSL certificates for secure HTTP connections.

    This function is called at the start of the test session to ensure
    proper SSL certificate handling for dataset downloads.
    """
    ssl_context = create_default_context(cafile=certifi.where())
    https_handler = HTTPSHandler(context=ssl_context)
    install_opener(build_opener(https_handler))


# =============================================================================
# Test Configuration Constants
# =============================================================================

CI = os.environ.get("CI") in ("true", "1")
HYPOTHESIS_ENABLED = os.environ.get("HYPOTHESIS_ENABLED") in (
    "true",
    "1",
)

# =============================================================================
# Hypothesis Configuration
# =============================================================================

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

# =============================================================================
# Pytest Hooks and Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options for pytest.

    This function adds three custom options to control test execution:
    - --run_stress: Run stress tests
    - --run_quality: Run quality tests
    - --run_unit: Run unit tests
    """
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
    """Modify test collection based on command line options.

    This function:
    1. Checks for hypothesis tests without examples
    2. Selectively skip tests based on selected categories (unit/quality/stress)
    """
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
    """Configure pytest settings and load hypothesis profiles.

    This function:
    1. Adds custom markers
    2. Records the available GPU memory
    3. Loads appropriate hypothesis profiles based on test execution context
    """
    config.addinivalue_line(
        "markers",
        "cudf_pandas: mark test as requiring the cudf.pandas wrapper",
    )
    cp.cuda.set_allocator(None)
    # max_gpu_memory: Capacity of the GPU memory in GB
    pytest.max_gpu_memory = _get_gpu_memory()
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
    """Skip tests that require the cudf.pandas accelerator.

    Tests marked with `@pytest.mark.cudf_pandas` will only be run if the
    cudf.pandas accelerator is enabled via the `cudf.pandas` plugin.
    """
    if "cudf_pandas" in pyfuncitem.keywords and not cudf.pandas.LOADED:
        pytest.skip("Test requires cudf.pandas accelerator")


def _get_gpu_memory():
    """Get the total GPU memory in GB."""
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


# =============================================================================
# Test Report and Logging Fixtures
# =============================================================================


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Create a test report and attach it to the test item.

    This function is used to create detailed test reports and attach them
    to the test items for later use in fixtures like failure_logger.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(scope="function")
def failure_logger(request):
    """Log the random seed used in failing tests.

    This fixture is used to log the random seed when a test fails,
    which helps in reproducing test failures.
    """
    yield
    if request.node.rep_call.failed:
        error_msg = " {} failed with seed: {}"
        error_msg = error_msg.format(
            request.node.nodeid, os.getenv("PYTEST_RANDOM_SEED")
        )
        print(error_msg)


# =============================================================================
# Random Seed Fixture
# =============================================================================


@pytest.fixture(scope="session")
def random_seed(request):
    """Generate or retrieve a random seed for test reproducibility.

    This fixture improves reproducibility of tests using random numbers
    by using the same seed throughout the test session.
    """
    current_random_seed = os.getenv("PYTEST_RANDOM_SEED")
    if current_random_seed is not None and current_random_seed.isdigit():
        random_seed = int(current_random_seed)
    else:
        random_seed = np.random.randint(0, 1e6)
        os.environ["PYTEST_RANDOM_SEED"] = str(random_seed)
    print("\nRandom seed value:", random_seed)
    return random_seed


# =============================================================================
# Dataset Fixtures
# =============================================================================


def dataset_fetch_retry(func, attempts=3, min_wait=1, max_wait=10):
    """Decorator for retrying dataset fetching operations with exponential backoff.

    This decorator implements retry logic for dataset fetching
    operations with exponential backoff. Wait times are in seconds.
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        reraise=True,
    )(func)


@pytest.fixture(scope="session")
def nlp_20news():
    """Load and preprocess the 20 newsgroups dataset.

    This fixture loads the 20 newsgroups dataset, preprocesses it using
    CountVectorizer, and returns the feature matrix and target vector.

    Returns
    -------
    tuple
        (X, Y) where X is the feature matrix and Y is the target vector
    """

    @dataset_fetch_retry
    def _fetch_20news():
        return fetch_20newsgroups(
            subset="train", shuffle=True, random_state=42
        )

    try:
        twenty_train = _fetch_20news()
    except Exception as e:
        pytest.xfail(f"Error fetching 20 newsgroup dataset: {str(e)}")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="session")
def housing_dataset():
    """Load and preprocess the California housing dataset.

    This fixture loads the California housing dataset and returns the
    feature matrix, target vector, and feature names.

    Returns
    -------
    tuple
        (X, y, feature_names) where X is the feature matrix, y is the target
        vector, and feature_names is a list of feature names
    """

    @dataset_fetch_retry
    def _fetch_housing():
        return fetch_california_housing()

    try:
        data = _fetch_housing()
    except Exception as e:
        pytest.xfail(f"Error fetching housing dataset: {str(e)}")

    X = cp.array(data["data"])
    y = cp.array(data["target"])
    feature_names = data["feature_names"]

    return X, y, feature_names


@pytest.fixture(scope="session")
def deprecated_boston_dataset():
    """Load and preprocess the deprecated Boston housing dataset.

    This fixture loads the Boston housing dataset from a GitHub URL since
    it was removed from scikit-learn. It returns the feature matrix and
    target vector.

    Note: This dataset is deprecated and should be replaced with a better
    alternative. See https://github.com/rapidsai/cuml/issues/5158

    Returns
    -------
    Bunch
        A Bunch object containing the data and target arrays
    """

    @dataset_fetch_retry
    def _get_boston_data():
        url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/datasets/data/boston_house_prices.csv"  # noqa: E501
        return pd.read_csv(url, header=None)

    try:
        df = _get_boston_data()
    except Exception as e:
        pytest.xfail(f"Error fetching Boston housing dataset: {str(e)}")

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
def supervised_learning_dataset(request, deprecated_boston_dataset):
    """Provide various supervised learning datasets for testing.

    This fixture provides access to multiple standard supervised learning
    datasets. It is parameterized to allow testing with different datasets.
    """
    datasets_dict = {
        "digits": datasets.load_digits(),
        "deprecated_boston_dataset": deprecated_boston_dataset,
        "diabetes": datasets.load_diabetes(),
        "cancer": datasets.load_breast_cancer(),
    }

    return datasets_dict[request.param].data


@pytest.fixture(scope="session")
def exact_shap_regression_dataset():
    """Generate a synthetic regression dataset for SHAP testing.

    This fixture creates a synthetic regression dataset with known
    properties for testing SHAP (SHapley Additive exPlanations) values.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) split of the synthetic dataset
    """
    X, y = with_dtype(
        make_regression(
            n_samples=101, n_features=11, noise=0.1, random_state=42
        ),
        np.float32,
    )
    return train_test_split(X, y, test_size=3, random_state=42)


@pytest.fixture(scope="session")
def exact_shap_classification_dataset():
    """Generate a synthetic classification dataset for SHAP testing.

    This fixture creates a synthetic classification dataset with known
    properties for testing SHAP (SHapley Additive exPlanations) values.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) split of the synthetic dataset
    """
    X, y = with_dtype(
        make_classification(
            n_samples=101, n_features=11, n_classes=2, random_state=42
        ),
        np.float32,
    )
    return train_test_split(X, y, test_size=3, random_state=42)
