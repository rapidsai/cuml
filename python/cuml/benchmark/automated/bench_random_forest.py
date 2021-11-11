import pytest
from .utils.utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#

"""
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_fil(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'FIL', 'classification', n_rows, n_features)
"""


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_rfc(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'RandomForestClassifier',
                    'classification', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_rfr(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'RandomForestRegressor',
                    'regression', n_rows, n_features)
