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


def bench_logistic_regression(
    gpubenchmark, bench_step, classification  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark, "LogisticRegression", bench_step, classification
    )


def bench_mbsgcclf(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "MBSGDClassifier", bench_step, classification
    )


def bench_knnclassifier(
    gpubenchmark, bench_step, classification  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark, "KNeighborsClassifier", bench_step, classification
    )


def bench_svc_linear(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(gpubenchmark, "SVC-Linear", bench_step, classification)


def bench_svc_rbf(gpubenchmark, bench_step, classification):  # noqa: F811
    _benchmark_algo(gpubenchmark, "SVC-RBF", bench_step, classification)


def bench_xgboost_classification(
    gpubenchmark, bench_step, classification  # noqa: F811
):
    pytest.importorskip("xgboost")
    _benchmark_algo(
        gpubenchmark, "xgboost-classification", bench_step, classification
    )
