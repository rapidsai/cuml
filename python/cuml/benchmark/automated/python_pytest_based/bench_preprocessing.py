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