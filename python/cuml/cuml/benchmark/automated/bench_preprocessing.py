#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

from .. import datagen
from .utils.utils import bench_step  # noqa: F401
from .utils.utils import _benchmark_algo

#
# Core tests
#


@pytest.fixture(scope="session")
def regression(request):
    dataset_kwargs = {
        "dataset_type": "regression",
        "n_samples": 10000,
        "n_features": 100,
    }
    dataset = datagen.gen_data(
        dataset_kwargs["dataset_type"],
        "cupy",
        n_samples=dataset_kwargs["n_samples"],
        n_features=dataset_kwargs["n_features"],
    )
    return dataset, dataset_kwargs


def bench_standardscaler(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(gpubenchmark, "StandardScaler", bench_step, regression)


def bench_maxabsscaler(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(gpubenchmark, "MaxAbsScaler", bench_step, regression)


def bench_normalizer(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(gpubenchmark, "Normalizer", bench_step, regression)
