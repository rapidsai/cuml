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
def blobs1(request):
    data = datagen.gen_data(
        "blobs",
        "cupy",
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
    )
    return data, {"dataset_type": "blobs", **request.param}


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


def bench_kmeans(gpubenchmark, bench_step, blobs1):  # noqa: F811
    _benchmark_algo(gpubenchmark, "KMeans", bench_step, blobs1)


@pytest.mark.parametrize(
    "algo_name",
    [
        "DBSCAN",
        "UMAP-Unsupervised",
        "UMAP-Supervised",
        "NearestNeighbors",
        "TSNE",
    ],
)
def bench_with_blobs(
    gpubenchmark,
    algo_name,
    bench_step,
    blobs2,  # noqa: F811
):
    # Lump together a bunch of simple blobs-based tests
    _benchmark_algo(gpubenchmark, algo_name, bench_step, blobs2)


@pytest.mark.parametrize("n_components", [2, 10, 50])
@pytest.mark.parametrize("algo_name", ["tSVD", "PCA"])
def bench_dimensionality_reduction(
    gpubenchmark,
    algo_name,
    bench_step,
    blobs3,
    n_components,  # noqa: F811
):
    _benchmark_algo(
        gpubenchmark,
        algo_name,
        bench_step,
        blobs3,
        setup_kwargs={"n_components": n_components},
    )
