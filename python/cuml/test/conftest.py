#
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

from math import ceil
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_classification as skl_make_clas
from sklearn.datasets import make_regression as skl_make_reg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def pytest_configure(config):
    cp.cuda.set_allocator(None)
    # max_gpu_memory: Capacity of the GPU memory in GB
    pytest.max_gpu_memory = get_gpu_memory()
    pytest.adapt_stress_test = 'CUML_ADAPT_STRESS_TESTS' in os.environ


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


def create_synthetic_dataset(generator=skl_make_reg,
                             n_samples=100,
                             n_features=10,
                             test_size=0.25,
                             random_state_generator=None,
                             random_state_train_test_split=None,
                             dtype=np.float32,
                             **kwargs):
    X, y = generator(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state_generator,
        **kwargs
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state_train_test_split
    )

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    return X_train, X_test, y_train, y_test


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
