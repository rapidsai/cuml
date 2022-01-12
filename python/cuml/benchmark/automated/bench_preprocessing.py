#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
