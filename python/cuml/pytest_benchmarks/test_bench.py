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
"""Demo integration of benchmarking to pytest interface
Requires pytest-benchmark, which is not currently installed by default.
"""

from cuml.benchmark import datagen, algorithms
from cuml.common.import_utils import has_pytest_benchmark
import pytest


#
# Testing utilities
#
def _benchmark_algo(
    benchmark,
    name,
    dataset_name,
    n_samples=10000,
    n_features=100,
    input_type='numpy',
    data_kwargs={},
    algo_args={},
):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset
    'dataset_name'"""
    algo = algorithms.algorithm_by_name(name)
    data = datagen.gen_data(
        dataset_name,
        input_type,
        n_samples=n_samples,
        n_features=n_features,
        **data_kwargs
    )

    def _benchmark_inner():
        algo.run_cuml(data, **algo_args)

    benchmark(_benchmark_inner)


#
# Core tests
#
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
def test_kmeans(benchmark, n_rows, n_features):
    _benchmark_algo(benchmark, 'KMeans', 'blobs', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('algo_name', ['DBSCAN', 'UMAP-Supervised',
                                       'NearestNeighbors'])
def test_with_blobs(benchmark, algo_name):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(benchmark, algo_name, 'blobs', 10000, 100)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_components', [2, 10, 50])
def test_pca(benchmark, n_components):
    _benchmark_algo(
        benchmark,
        'PCA',
        'blobs',
        50000,
        100,
        algo_args=dict(n_components=n_components),
    )
