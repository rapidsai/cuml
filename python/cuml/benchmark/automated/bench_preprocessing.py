import pytest
from .utils.utils import _benchmark_algo
from cuml.common.import_utils import has_pytest_benchmark

#
# Core tests
#


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.ML
def bench_standardscaler(gpubenchmark):
    _benchmark_algo(gpubenchmark, 'StandardScaler', 'regression')


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.ML
def bench_maxabsscaler(gpubenchmark):
    _benchmark_algo(gpubenchmark, 'MaxAbsScaler', 'regression')


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.ML
def bench_normalizer(gpubenchmark):
    _benchmark_algo(gpubenchmark, 'Normalizer', 'regression')
