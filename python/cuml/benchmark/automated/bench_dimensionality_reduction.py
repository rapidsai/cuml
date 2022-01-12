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
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_kmeans(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'KMeans', 'blobs', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('algo_name', ['DBSCAN',
                                       'UMAP-Unsupervised',
                                       'UMAP-Supervised',
                                       'NearestNeighbors',
                                       'TSNE'
                                       ])
@pytest.mark.ML
def bench_with_blobs(gpubenchmark, algo_name):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(gpubenchmark, algo_name, 'blobs', 10000, 100)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_components', [2, 10, 50])
@pytest.mark.parametrize('algo_name', ['tSVD',
                                       'PCA'])
@pytest.mark.ML
def bench_dimensionality_reduction(gpubenchmark, n_components, algo_name):
    _benchmark_algo(
        gpubenchmark,
        algo_name,
        'blobs',
        50000,
        100,
        algo_args=dict(n_components=n_components),
    )
