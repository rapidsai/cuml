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
def bench_logistic_regression(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'LogisticRegression', 'classification', n_rows, n_features)
    
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_mbsgcclf(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'MBSGDClassifier', 'classification', n_rows, n_features)
    

@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_knnclassifier(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'KNeighborsClassifier', 'classification', n_rows, n_features)

@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_knnclassifier(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'SVC-Linear', 'classification', n_rows, n_features)


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_svc_rbf(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'SVC-RBF', 'classification', n_rows, n_features)

    
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_rows', [1000, 10000])
@pytest.mark.parametrize('n_features', [5, 500])
@pytest.mark.ML
def bench_svc_linear(gpubenchmark, n_rows, n_features):
    _benchmark_algo(gpubenchmark, 'SVC-Linear', 'classification', n_rows, n_features)