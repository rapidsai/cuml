import pytest
from ..init_pytest_bench import *  # noqa: F401,F403
from ..utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mnmg_kmeans(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(gpubenchmark, 'MNMG.KMeans',
                    'blobs', n_rows, n_features, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('algo_name', ['MNMG.DBSCAN',
                                       'MNMG.UMAP-Unsupervised',
                                       'MNMG.UMAP-Supervised',
                                       'MNMG.NearestNeighbors'
                                       ])
@pytest.mark.ML
def bench_mnmg_with_blobs(gpubenchmark, algo_name, client):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(gpubenchmark, algo_name,
                    'blobs', 10000, 100, client=client)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_components', [2, 10, 50])
@pytest.mark.parametrize('algo_name', ['MNMG.tSVD',
                                       'MNMG.PCA'])
@pytest.mark.ML
def bench_mnmg_dimensionality_reduction(gpubenchmark, n_components,
                                        algo_name, client):
    _benchmark_algo(
        gpubenchmark,
        algo_name,
        'blobs',
        50000,
        100,
        algo_args=dict(n_components=n_components),
        client=client)
