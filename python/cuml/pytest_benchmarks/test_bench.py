from cuml import benchmark
from cuml.benchmark import bench_data, bench_algos
import pytest

#
# Testing utilities
#
def _benchmark_algo(benchmark, name, dataset_name, n_samples=10000,
                    n_features=100, input_type='numpy', data_kwargs={}, algo_kwargs={}):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset 'dataset_name'"""
    algo = bench_algos.algorithm_by_name(name)
    data = bench_data.gen_data(dataset_name, input_type, n_samples=n_samples,
                               n_features=n_features, **data_kwargs)

    def _benchmark_inner():
        algo.run_cuml(data, **algo_kwargs)

    benchmark(_benchmark_inner)

#
# Core tests
#
@pytest.mark.parametrize('nrows', [1000, 10000])
@pytest.mark.parametrize('nfeatures', [5, 500])
def test_kmeans(benchmark, nrows, nfeatures):
    _benchmark_algo(benchmark, 'KMeans', 'blobs', nrows, nfeatures)

def test_dbscan(benchmark):
    _benchmark_algo(benchmark, 'DBSCAN', 'blobs', 10000, 100)

def test_umap(benchmark):
    _benchmark_algo(benchmark, 'UMAP', 'blobs', 10000, 100)

def test_nearest_neighbors(benchmark):
    _benchmark_algo(benchmark, 'NearestNeighbors', 'blobs', 10000, 100)
