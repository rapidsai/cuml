#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from .. import datagen
from .utils.utils import bench_step  # noqa: F401
from .utils.utils import _benchmark_algo, fixture_generation_helper

#
# Core tests
#


@pytest.fixture(
    **fixture_generation_helper(
        {"n_samples": [1000, 10000], "n_features": [5, 500]}
    )
)
def classification(request):
    data = datagen.gen_data(
        "classification",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, {"dataset_type": "classification", **request.param}


@pytest.fixture(
    **fixture_generation_helper(
        {"n_samples": [1000, 10000], "n_features": [5, 500]}
    )
)
def regression(request):
    data = datagen.gen_data(
        "regression",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, {"dataset_type": "regression", **request.param}


"""
def bench_fil(gpubenchmark, bench_step, classification):
    _benchmark_algo(gpubenchmark, 'FIL',
                    bench_step, classification)
"""


def bench_rfc(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "RandomForestClassifier", bench_step, classification
    )


def bench_rfr(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "RandomForestRegressor", bench_step, regression
    )
