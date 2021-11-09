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
def bench_mnmg_knnclassifier(gpubenchmark, n_rows, n_features, client):
    _benchmark_algo(gpubenchmark, 'MNMG.KNeighborsClassifier',
                    'classification', n_rows, n_features, client=client)
