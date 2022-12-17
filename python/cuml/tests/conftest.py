#
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

import pytest
import os
import subprocess

import numpy as np
import cupy as cp
import hypothesis

from math import ceil
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_classification as skl_make_clas
from sklearn.datasets import make_regression as skl_make_reg
from sklearn.feature_extraction.text import CountVectorizer
from cuml.testing.utils import create_synthetic_dataset


# Add the import here for any plugins that should be loaded EVERY TIME
pytest_plugins = ("cuml.testing.plugins.quick_run_plugin")

CI = os.environ.get("CI") in ("true", "1")


# Configure hypothesis profiles

HEALTH_CHECKS_SUPPRESSED_BY_DEFAULT = \
    hypothesis.HealthCheck.all() if CI else [
        hypothesis.HealthCheck.data_too_large,
        hypothesis.HealthCheck.too_slow,
    ]

hypothesis.settings.register_profile(
    name="unit",
    parent=hypothesis.settings.get_profile("default"),
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
    max_examples=200
)


def pytest_addoption(parser):
    # Any custom option, that should be available at any time (not just a
    # plugin), goes here.

    group = parser.getgroup('cuML Custom Options')

    group.addoption(
        "--run_stress",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'stress'. These are the most intense "
              "tests that take the longest to run and are designed to stress "
              "the hardware's compute resources."))

    group.addoption(
        "--run_quality",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'quality'. These tests are more "
              "computationally intense than 'unit', but less than 'stress'"))

    group.addoption(
        "--run_unit",
        action="store_true",
        default=False,
        help=("Runs tests marked with 'unit'. These are the quickest tests "
              "that are focused on accuracy and correctness."))


def pytest_collection_modifyitems(config, items):

    should_run_quality = config.getoption("--run_quality")
    should_run_stress = config.getoption("--run_stress")

    # Run unit is implied if no --run_XXX is set
    should_run_unit = config.getoption("--run_unit") or not (
        should_run_quality or should_run_stress)

    # Mark the tests as "skip" if needed
    if not should_run_unit:
        skip_unit = pytest.mark.skip(
            reason="Unit tests run with --run_unit flag.")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)

    if not should_run_quality:
        skip_quality = pytest.mark.skip(
            reason="Quality tests run with --run_quality flag.")
        for item in items:
            if "quality" in item.keywords:
                item.add_marker(skip_quality)

    if not should_run_stress:
        skip_stress = pytest.mark.skip(
            reason="Stress tests run with --run_stress flag.")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)


def pytest_configure(config):
    cp.cuda.set_allocator(None)
    # max_gpu_memory: Capacity of the GPU memory in GB
    pytest.max_gpu_memory = get_gpu_memory()
    pytest.adapt_stress_test = 'CUML_ADAPT_STRESS_TESTS' in os.environ

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


@pytest.fixture(scope="module")
def nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(subset='train',
                                          shuffle=True,
                                          random_state=42)
    except:  # noqa E722
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="module")
def housing_dataset():
    try:
        data = fetch_california_housing()

    # failing to download has appeared as multiple varied errors in CI
    except:  # noqa E722
        pytest.xfail(reason="Error fetching housing dataset")

    X = cp.array(data['data'])
    y = cp.array(data['target'])

    feature_names = data['feature_names']

    return X, y, feature_names


@pytest.fixture(scope="session")
def random_seed(request):
    current_random_seed = os.getenv('PYTEST_RANDOM_SEED')
    if current_random_seed is not None and current_random_seed.isdigit():
        random_seed = int(current_random_seed)
    else:
        random_seed = np.random.randint(0, 1e6)
        os.environ['PYTEST_RANDOM_SEED'] = str(random_seed)
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
        error_msg = error_msg.format(request.node.nodeid,
                                     os.getenv('PYTEST_RANDOM_SEED'))
        print(error_msg)


@pytest.fixture(scope="module")
def exact_shap_regression_dataset():
    return create_synthetic_dataset(generator=skl_make_reg,
                                    n_samples=101,
                                    n_features=11,
                                    test_size=3,
                                    random_state_generator=42,
                                    random_state_train_test_split=42,
                                    noise=0.1)


@pytest.fixture(scope="module")
def exact_shap_classification_dataset():
    return create_synthetic_dataset(generator=skl_make_clas,
                                    n_samples=101,
                                    n_features=11,
                                    test_size=3,
                                    random_state_generator=42,
                                    random_state_train_test_split=42)


def get_gpu_memory():
    bash_command = "nvidia-smi --query-gpu=memory.total --format=csv"
    output = subprocess.check_output(bash_command,
                                     shell=True).decode("utf-8")
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
