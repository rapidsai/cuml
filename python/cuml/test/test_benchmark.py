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
import cuml
from cuml import benchmark
from cuml.benchmark import bench_data
from cuml.benchmark.bench_runners import AccuracyComparisonRunner
import numpy as np
import cudf
import pytest
from numba import cuda
from sklearn import metrics

@pytest.mark.parametrize('dataset', ['blobs', 'regression', 'classification'])
def test_data_generators(dataset):
    data = bench_data.gen_data(dataset, "numpy",
                               n_samples=100, n_features=10)
    assert isinstance(data, np.array)
    assert data[0].shape[0] == 100


@pytest.mark.parametrize('input_type', ['numpy', 'cudf'])
def test_data_generators(input_type):
    X, *_  = bench_data.gen_data('blobs', input_type,
                              n_samples=100, n_features=10)
    if input_type == 'numpy':
        assert isinstance(X, np.ndarray)
    elif input_type == 'cudf':
        assert isinstance(X, cudf.DataFrame)
    elif input_type == 'pandas':
        assert isinstance(X, pd.DataFrame)
    elif input_type == 'gpuarray':
        assert cuda.is_cuda_array(X)
    else:
        assert False


def test_data_generator_split():
    X_train, y_train, X_test, y_test  = bench_data.gen_data('blobs',
                                                            'numpy',
                                                            n_samples=100,
                                                            n_features=10,
                                                            test_fraction=0.20)
    assert X_train.shape == (100,10)
    assert X_test.shape == (25,10)

def test_run_variations():
    from cuml.benchmark.bench_all import run_variations
    pass

def test_accuracy_runner(mocker):
    # Generate a trivial data pattern to test
    from cuml.benchmark.bench_algos import AlgorithmPair

    mocker.patch('cuml.benchmark.bench_data.gen_data').return_value = (
        np.zeros((5,2)), np.zeros(5),
        np.zeros((5,2)), np.ones(5) )
    class MockAlgo:
        def fit(self, X, y):
            return
        def predict(self, X):
            return np.array([0,0,1,0,0])

    pair = AlgorithmPair(MockAlgo,
                         MockAlgo,
                         shared_args={},
                         name="Mock",
                         accuracy_function=metrics.accuracy_score)



    runner = AccuracyComparisonRunner([100], [10])
    runner.run(pair)
