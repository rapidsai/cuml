# Copyright (c) 2019, NVIDIA CORPORATION.
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
from cuml.benchmark import datagen, algorithms
from cuml.benchmark.bench_helper_funcs import _training_data_to_numpy
from cuml.benchmark.runners import AccuracyComparisonRunner, \
    SpeedupComparisonRunner, run_variations

import numpy as np
import cudf
import pytest
from numba import cuda
from sklearn import metrics
import pandas as pd

import time


@pytest.mark.parametrize('dataset', ['blobs', 'regression', 'classification'])
def test_data_generators(dataset):
    data = datagen.gen_data(dataset, "numpy", n_samples=100, n_features=10)
    assert isinstance(data[0], np.ndarray)
    assert data[0].shape[0] == 100


@pytest.mark.parametrize('input_type',
                         ['numpy', 'cudf', 'pandas', 'gpuarray', 'gpuarray-c'])
def test_data_generator_types(input_type):
    X, *_ = datagen.gen_data('blobs', input_type, n_samples=100, n_features=10)
    if input_type == 'numpy':
        assert isinstance(X, np.ndarray)
    elif input_type == 'cudf':
        assert isinstance(X, cudf.DataFrame)
    elif input_type == 'pandas':
        assert isinstance(X, pd.DataFrame)
    elif input_type == 'gpuarray':
        assert cuda.is_cuda_array(X)
    elif input_type == 'gpuarray-c':
        assert cuda.is_cuda_array(X)
    else:
        assert False


def test_data_generator_split():
    X_train, y_train, X_test, y_test = datagen.gen_data(
        'blobs', 'numpy', n_samples=100, n_features=10, test_fraction=0.20
    )
    assert X_train.shape == (100, 10)
    assert X_test.shape == (25, 10)


def test_run_variations():
    algo = algorithms.algorithm_by_name("LogisticRegression")

    res = run_variations(
        [algo],
        dataset_name="classification",
        bench_rows=[100, 200],
        bench_dims=[10, 20],
    )
    assert res.shape[0] == 4
    assert (res.n_samples == 100).sum() == 2
    assert (res.n_features == 20).sum() == 2


def test_speedup_runner():
    class MockAlgo:
        def __init__(self, t):
            self.t = t

        def fit(self, X, y):
            time.sleep(self.t)
            return

        def predict(self, X):
            nr = X.shape[0]
            res = np.zeros(nr)
            res[0:int(nr / 5.0)] = 1.0
            return res

    class FastMockAlgo(MockAlgo):
        def __init__(self):
            MockAlgo.__init__(self, 0.1)

    class SlowMockAlgo(MockAlgo):
        def __init__(self):
            MockAlgo.__init__(self, 2)

    pair = algorithms.AlgorithmPair(
        SlowMockAlgo,
        FastMockAlgo,
        shared_args={},
        name="Mock",
        accuracy_function=metrics.accuracy_score,
    )

    runner = SpeedupComparisonRunner(
        [20], [5], dataset_name='zeros'
    )
    results = runner.run(pair)[0]

    expected_speedup = SlowMockAlgo().t / FastMockAlgo().t

    assert results["speedup"] == pytest.approx(expected_speedup, 0.4)


def test_multi_reps():
    class CountingAlgo:
        tot_reps = 0

        def fit(self, X, y):
            CountingAlgo.tot_reps += 1

    pair = algorithms.AlgorithmPair(
        CountingAlgo,
        CountingAlgo,
        shared_args={},
        name="Counting",
    )

    runner = AccuracyComparisonRunner(
        [20], [5], dataset_name='zeros', test_fraction=0.20, n_reps=4
    )
    runner.run(pair)

    # Double the n_reps since it is used in cpu and cuml versions
    assert CountingAlgo.tot_reps == 8


def test_accuracy_runner():
    # Set up data that should deliver accuracy of 0.20 if all goes right
    class MockAlgo:
        def fit(self, X, y):
            return

        def predict(self, X):
            nr = X.shape[0]
            res = np.zeros(nr)
            res[0:int(nr / 5.0)] = 1.0
            return res

    pair = algorithms.AlgorithmPair(
        MockAlgo,
        MockAlgo,
        shared_args={},
        name="Mock",
        accuracy_function=metrics.accuracy_score,
    )

    runner = AccuracyComparisonRunner(
        [20], [5], dataset_name='zeros', test_fraction=0.20
    )
    results = runner.run(pair)[0]

    assert results["cuml_acc"] == pytest.approx(0.80)


@pytest.mark.parametrize('input_type', ['numpy', 'cudf', 'pandas', 'gpuarray'])
def test_training_data_to_numpy(input_type):
    X, y, *_ = datagen.gen_data(
        'blobs', input_type, n_samples=100, n_features=10
    )
    X_np, y_np = _training_data_to_numpy(X, y)
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
