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
    **fixture_generation_helper(
        {"n_samples": [1000, 10000], "n_features": [5, 500]}
    )
)
def blobs1(request):
    data = datagen.gen_data(
        "classification",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, None


@pytest.fixture(scope="session")
def blobs2(request):
    dataset_kwargs = {
        "dataset_type": "blobs",
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


@pytest.fixture(scope="session")
def blobs3(request):
    dataset_kwargs = {
        "dataset_type": "blobs",
        "n_samples": 50000,
        "n_features": 100,
    }
    dataset = datagen.gen_data(
        dataset_kwargs["dataset_type"],
        "cupy",
        n_samples=dataset_kwargs["n_samples"],
        n_features=dataset_kwargs["n_features"],
    )
    return dataset, dataset_kwargs


def bench_mnmg_kmeans(gpubenchmark, bench_step, blobs1, client):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "MNMG.KMeans", bench_step, blobs1, client=client
    )


def bench_mnmg_dbscan(gpubenchmark, bench_step, blobs2, client):  # noqa: F811
    _benchmark_algo(
        gpubenchmark, "MNMG.DBSCAN", bench_step, blobs2, client=client
    )


def bench_mnmg_nearest_neighbors(
    gpubenchmark, bench_step, blobs2, client  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark,
        "MNMG.NearestNeighbors",
        bench_step,
        blobs2,
        client=client,
    )


@pytest.mark.parametrize(
    "algo_name", ["MNMG.UMAP-Unsupervised", "MNMG.UMAP-Supervised"]
)
def bench_mnmg_umap(
    gpubenchmark, algo_name, bench_step, blobs2, client  # noqa: F811
):
    _benchmark_algo(gpubenchmark, algo_name, bench_step, blobs2, client=client)


@pytest.mark.parametrize("algo_name", ["MNMG.tSVD", "MNMG.PCA"])
@pytest.mark.parametrize("n_components", [2, 10, 50])
def bench_mnmg_dimensionality_reduction(
    gpubenchmark,
    algo_name,
    bench_step,
    blobs3,  # noqa: F811
    client,
    n_components,
):
    _benchmark_algo(
        gpubenchmark,
        algo_name,
        bench_step,
        blobs3,
        setup_kwargs={"n_components": n_components},
        client=client,
    )
