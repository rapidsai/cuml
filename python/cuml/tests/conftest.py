#
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import os
from datetime import timedelta
from math import ceil

import cudf.pandas
import cupy as cp
import hypothesis
import numpy as np
import pynvml
import pytest
from scipy import sparse
from sklearn import datasets

# =============================================================================
# Pytest Configuration
# =============================================================================

# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = "cuml.testing.plugins.quick_run_plugin"


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
    - --run_memleak: Run memleak tests
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

    group.addoption(
        "--run_memleak",
        action="store_true",
        default=False,
        help="Runs tests marked with 'memleak'. These test for memory leaks.",
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
    should_run_memleak = config.getoption("--run_memleak")

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

    if not should_run_memleak:
        skip_memleak = pytest.mark.skip(
            reason="Memory leak tests run with --run_memleak flag."
        )
        for item in items:
            if "memleak" in item.keywords:
                item.add_marker(skip_memleak)


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


def _get_pynvml_device_handle(device_id=0):
    """Get GPU handle from device index or UUID.

    Parameters
    ----------
    device_id: int or str
        The index or UUID of the device from which to obtain the handle.

    Raises
    ------
    ValueError
        If acquiring the device handle for the device specified failed.
    pynvml.NVMLError
        If any NVML error occurred while initializing.

    Returns
    -------
    A pynvml handle to the device.

    Examples
    --------
    >>> _get_pynvml_device_handle(device_id=0)

    >>> _get_pynvml_device_handle(device_id="GPU-9fb42d6f-7d6b-368f-f79c-3c3e784c93f6")
    """
    pynvml.nvmlInit()

    try:
        if device_id and not str(device_id).isnumeric():
            # This means device_id is UUID.
            # This works for both MIG and non-MIG device UUIDs.
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(device_id))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                # Additionally get parent device handle
                # if the device itself is a MIG instance
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
                    handle
                )
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return handle
    except pynvml.NVMLError:
        raise ValueError(f"Invalid device index or UUID: {device_id}")


def _get_gpu_memory(device_index=0):
    """Return total memory of CUDA device with index or with device identifier UUID.

    Parameters
    ----------
    device_index: int or str
        The index or UUID of the device from which to obtain the CPU affinity.

    Returns
    -------
    The total memory of the CUDA Device in GB, or ``None`` for devices that do not
    have a dedicated memory resource, as is usually the case for system on a chip (SoC)
    devices.
    """
    handle = _get_pynvml_device_handle(device_index)

    try:
        # Return total memory in GB
        return ceil(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 2**30)
    except pynvml.NVMLError_NotSupported:
        return None


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


@pytest.fixture(scope="session")
def nlp_20news():
    """Generate a sparse text-like dataset similar to 20 newsgroups.

    This fixture generates a sparse bag-of-words matrix and target vector
    that mimic the characteristics of the 20 newsgroups dataset after
    CountVectorizer/TF-IDF transformation, using topic-like word distributions.

    Returns
    -------
    tuple
        (X, Y) where X is a sparse feature matrix and Y is the target vector
    """
    n_docs = 11314  # Similar to 20 newsgroups training set
    n_features = 10000  # Vocabulary size
    n_classes = 20
    avg_nonzero_per_doc = 150

    rng = np.random.RandomState(0)

    # Class labels (balanced)
    y = rng.randint(0, n_classes, size=n_docs)

    # Class-specific word distributions (topic-like)
    # Each class has its own word preference distribution over the vocabulary
    class_word_probs = []
    for _ in range(n_classes):
        # Dirichlet distribution to simulate "topic" structure with sparsity
        alpha = np.ones(n_features) * 0.01
        topic = rng.dirichlet(alpha)
        class_word_probs.append(topic)
    class_word_probs = np.vstack(class_word_probs)

    # Generate sparse bag-of-words for each document
    data = []
    rows = []
    cols = []

    for i in range(n_docs):
        label = y[i]
        # Document length ~ Poisson around avg_nonzero_per_doc
        doc_len = max(1, rng.poisson(avg_nonzero_per_doc))
        # Sample word indices from the class distribution
        word_indices = rng.choice(
            n_features,
            size=doc_len,
            replace=True,
            p=class_word_probs[label],
        )
        # Count word occurrences
        unique, counts = np.unique(word_indices, return_counts=True)
        rows.extend([i] * len(unique))
        cols.extend(unique.tolist())
        data.extend(counts.tolist())

    X_bow = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_docs, n_features),
        dtype=np.float64,
    )

    # Apply TF-IDF-like weighting
    df = (X_bow > 0).sum(axis=0).A1 + 1.0  # document frequency + 1
    idf = np.log((1.0 + n_docs) / df)
    X = X_bow.multiply(idf).tocsr()

    Y = cp.array(y)

    return X, Y


@pytest.fixture(scope="session")
def housing_dataset():
    """Generate a regression dataset similar to California housing.

    This fixture generates a regression dataset that mimics the
    characteristics of the California housing dataset.

    Returns
    -------
    tuple
        (X, y) where X is the feature matrix and y is the target vector
    """
    n_samples = 20640  # Same as California housing
    n_features = 8

    X, y = datasets.make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.5,
        random_state=42,
    )

    X = cp.array(X)
    y = cp.array(y)

    return X, y


@pytest.fixture(
    scope="session",
    params=["digits", "diabetes", "cancer"],
)
def supervised_learning_dataset(request):
    """Provide various supervised learning datasets for testing.

    This fixture provides access to multiple standard supervised learning
    datasets. It is parameterized to allow testing with different datasets.
    """
    datasets_dict = {
        "digits": datasets.load_digits(),
        "diabetes": datasets.load_diabetes(),
        "cancer": datasets.load_breast_cancer(),
    }

    return datasets_dict[request.param].data
