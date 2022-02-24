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
from .utils.utils import _benchmark_algo, fixture_generation_helper
from .utils.utils import bench_step  # noqa: F401
from .. import datagen

#
# Core tests
#


@pytest.fixture(**fixture_generation_helper({
                    'n_samples': [1000, 10000],
                    'n_features': [5, 500]
                }))
def classification(request):
    data = datagen.gen_data(
        'classification',
        'cupy',
        n_samples=request.param['n_samples'],
        n_features=request.param['n_features']
    )
    return data, {
                    'dataset_type': 'classification',
                    **request.param
                 }


@pytest.fixture(**fixture_generation_helper({
                    'n_samples': [1000, 10000],
                    'n_features': [5, 500]
                }))
def regression(request):
    data = datagen.gen_data(
        'regression',
        'cupy',
        n_samples=request.param['n_samples'],
        n_features=request.param['n_features']
    )
    return data, {
                    'dataset_type': 'regression',
                    **request.param
                 }


"""
def bench_fil(gpubenchmark, bench_step, classification):
    _benchmark_algo(gpubenchmark, 'FIL',
                    bench_step, classification)
"""


def bench_rfc(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(gpubenchmark, 'RandomForestClassifier',
                    bench_step, classification)


def bench_rfr(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(gpubenchmark, 'RandomForestRegressor',
                    bench_step, regression)
