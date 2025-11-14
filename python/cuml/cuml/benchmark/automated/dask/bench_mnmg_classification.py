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
    **fixture_generation_helper(
        {"n_samples": [1000, 10000], "n_features": [5, 500]}
    )
)
def classification(request):
    data = datagen.gen_data(
        "classification",
        "cudf",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, None


def bench_mnmg_knnclassifier(
    gpubenchmark,
    bench_step,
    classification,
    client,  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark,
        "MNMG.KNeighborsClassifier",
        bench_step,
        classification,
        client=client,
    )
