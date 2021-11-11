#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
from ..utils.utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_linear_regression(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(gpubenchmark, 'MNMG.LinearRegression',
                    'regression', n_rows, n_features, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mnmg_lasso(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(
        gpubenchmark,
        'MNMG.Lasso',
        'regression',
        n_rows, n_features, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mnmg_elastic(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(
        gpubenchmark,
        'MNMG.ElasticNet',
        'regression',
        n_rows, n_features, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mnmg_ridge(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(
        gpubenchmark,
        'MNMG.Ridge',
        'regression',
        n_rows, n_features, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mnmg_knnregressor(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(
        gpubenchmark,
        'MNMG.KNeighborsRegressor',
        'regression',
        n_rows, n_features, client=client)
