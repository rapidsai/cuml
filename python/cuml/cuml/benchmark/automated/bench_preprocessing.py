#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
