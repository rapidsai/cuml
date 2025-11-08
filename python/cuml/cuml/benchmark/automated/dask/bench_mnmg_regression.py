#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ... import datagen
from ..utils.utils import bench_step  # noqa: F401
from ..utils.utils import _benchmark_algo, fixture_generation_helper

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
