#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from .. import datagen
from .utils.utils import bench_step  # noqa: F401
from .utils.utils import _benchmark_algo, fixture_generation_helper


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


def bench_etc(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "ExtraTreesClassifier", bench_step, classification
    )


def bench_etr(gpubenchmark, bench_step, regression):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "ExtraTreesRegressor", bench_step, regression
    )
