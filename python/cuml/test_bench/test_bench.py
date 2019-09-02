from cuml import benchmark
from cuml.benchmark import bench_data, bench_algos
import pytest

#
# Testing utilities
#
def _benchmark_algo(benchmark, name, dataset_name, n_samples=10000, n_features=100, input_type='numpy'):
    algo = bench_algos.algorithm_by_name(name)
    data = bench_data.gen_data(dataset_name, input_type, n_samples=n_samples, n_features=n_features)

    print(data[0].__class__)

    def _benchmark_inner():
        algo.run_cuml(data)

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

