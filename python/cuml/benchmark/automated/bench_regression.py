#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from .utils.utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 300])
@pytest.mark.ML
def bench_linear_regression(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'LinearRegression',
                    'regression', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_lasso(gpubenchmark, n_rows, n_features):
    _benchmark_algo(
        gpubenchmark,
        'Lasso',
        'regression',
        n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_elastic(gpubenchmark, n_rows, n_features):
    _benchmark_algo(
        gpubenchmark,
        'ElasticNet',
        'regression',
        n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_ridge(gpubenchmark, n_rows, n_features):
    _benchmark_algo(
        gpubenchmark,
        'Ridge',
        'regression',
        n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_knnregressor(gpubenchmark, n_rows, n_features):
    _benchmark_algo(
        gpubenchmark,
        'KNeighborsRegressor',
        'regression',
        n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_svr_rbf(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'SVR-RBF',
                    'regression', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [500, 4000])
@pytest.mark.parametrize('n_features', [5, 400])
@pytest.mark.ML
def bench_svr_linear(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'SVR-Linear',
                    'regression', n_rows, n_features)
