import pytest
from .init_pytest_bench import *  # noqa: F401,F403
from .utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
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
