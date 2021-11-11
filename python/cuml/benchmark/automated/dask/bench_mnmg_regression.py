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
