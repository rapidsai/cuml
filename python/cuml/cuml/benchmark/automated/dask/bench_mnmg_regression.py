#
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
from ..utils.utils import _benchmark_algo, fixture_generation_helper
from ..utils.utils import bench_step  # noqa: F401
from ... import datagen

#
# Core tests
#


@pytest.fixture(
    **fixture_generation_helper({"n_samples": [10000], "n_features": [5, 500]})
)
def regression(request):
    data = datagen.gen_data(
        "regression",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, None


def bench_linear_regression(
    gpubenchmark, bench_step, regression, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark,
        "MNMG.LinearRegression",
        bench_step,
        regression,
        client=client,
    )


def bench_mnmg_lasso(
    gpubenchmark, bench_step, regression, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark, "MNMG.Lasso", bench_step, regression, client=client
    )


def bench_mnmg_elastic(
    gpubenchmark, bench_step, regression, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark, "MNMG.ElasticNet", bench_step, regression, client=client
    )


def bench_mnmg_ridge(
    gpubenchmark, bench_step, regression, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark, "MNMG.Ridge", bench_step, regression, client=client
    )


def bench_mnmg_knnregressor(
    gpubenchmark, bench_step, regression, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark,
        "MNMG.KNeighborsRegressor",
        bench_step,
        regression,
        client=client,
    )
