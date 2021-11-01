import pytest

import pytest_benchmark
# FIXME: Remove this when rapids_pytest_benchmark.gpubenchmark is available
# everywhere
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
from cuml.datasets import make_classification, make_blobs, make_regression
from cuml.benchmark import datagen, algorithms
from cuml.common.import_utils import has_pytest_benchmark
import pytest


########
#Helpers
@pytest.fixture(scope="module", params=([1000,10000]))
def regressionData(request):
    return make_regression(request.param, n_features=15)

@pytest.fixture(scope="module", params=([1000,10000]))
def clfData(request):
    return make_classification(request.param, n_features=15)

@pytest.fixture(scope="module", params=([1000,10000]))
def blobData(request):
    return make_blobs(request.param, n_features=15)

# Record the current RMM settings so reinitialize() will be called only when a
# change is needed (RMM defaults both values to False). This allows the
# --no-rmm-reinit option to prevent reinitialize() from being called at all
# (see conftest.py for details).
RMM_SETTINGS = {"managed_mem": False,
                "pool_alloc": False}


def reinitRMM(managed_mem, pool_alloc):

    if (managed_mem != RMM_SETTINGS["managed_mem"]) or \
       (pool_alloc != RMM_SETTINGS["pool_alloc"]):

        rmm.reinitialize(
            managed_memory=managed_mem,
            pool_allocator=pool_alloc,
            initial_pool_size=2 << 27
        )
        RMM_SETTINGS.update(managed_mem=managed_mem,
                            pool_alloc=pool_alloc)


# @pytest.mark.ML
# def bench_linear_regression(gpubenchmark, regressionData):
#     mod = cuml.linear_model.LinearRegression()
#     gpubenchmark(mod.fit,
#                  regressionData[0],
#                  regressionData[1])

# @pytest.mark.ML
# def bench_ridge_regression(gpubenchmark, regressionData):
#     mod = cuml.Ridge()
#     gpubenchmark(mod.fit,
#                  regressionData[0],
#                  regressionData[1])
    
# @pytest.mark.ML
# def bench_logistic(gpubenchmark, clfData):
#     mod = cuml.linear_model.LogisticRegression()
#     gpubenchmark(mod.fit,
#                  clfData[0],
#                  clfData[1])

# @pytest.mark.ML
# def bench_agglomerative(gpubenchmark, blobData):
#     mod = cuml.cluster.AgglomerativeClustering()
#     gpubenchmark(mod.fit,
#                  blobData[0])

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
@pytest.mark.ML
def test_kmeans(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'KMeans', 'blobs', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('algo_name', ['DBSCAN', 'UMAP-Supervised',
                                       'NearestNeighbors'])
@pytest.mark.ML
def test_with_blobs(gpubenchmark, algo_name):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(gpubenchmark, algo_name, 'blobs', 10000, 100)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_components', [2, 10, 50])
@pytest.mark.ML
def test_pca(gpubenchmark, n_components):
    _benchmark_algo(
        gpubenchmark,
        'PCA',
        'blobs',
        50000,
        100,
        algo_args=dict(n_components=n_components),
    )