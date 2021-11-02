import pytest
from .utils import _benchmark_algo
try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
          "falling back to pytest_benchmark fixtures.\n")

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass

import cuml
import rmm
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def test_kmeans(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'KMeans', 'blobs', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('algo_name', ['DBSCAN',
                                       'UMAP-Supervised',
                                       'UMAP-Supervised',
                                       'NearestNeighbors',
                                       'TSNE'
                                       ])
@pytest.mark.ML
def test_with_blobs(gpubenchmark, algo_name):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(gpubenchmark, algo_name, 'blobs', 10000, 100)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_components', [2, 10, 50])
@pytest.mark.parametrize('algo_name', ['tSVD',
                                       'PCA'
                                      ])
@pytest.mark.ML
def test_dimensionality_reduction(gpubenchmark, n_components, algo_name):
    _benchmark_algo(
        gpubenchmark,
        algo_name,
        'blobs',
        50000,
        100,
        algo_args=dict(n_components=n_components),
    )