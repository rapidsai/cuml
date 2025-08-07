# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from cuml import DBSCAN as cuDBSCAN
from cuml.testing.datasets import make_pattern
from cuml.testing.utils import (
    array_equal,
    assert_dbscan_equal,
    get_handle,
    quality_param,
    stress_param,
    unit_param,
)


@pytest.mark.parametrize("max_mbytes_per_batch", [1e3, None])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "ncols", [unit_param(20), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize(
    "out_dtype",
    [
        unit_param("int32"),
        unit_param(np.int32),
        unit_param("int64"),
        unit_param(np.int64),
        quality_param("int32"),
        stress_param("int32"),
    ],
)
@pytest.mark.parametrize("algorithm", ["brute", "rbc"])
def test_dbscan(
    datatype,
    use_handle,
    nrows,
    ncols,
    max_mbytes_per_batch,
    out_dtype,
    algorithm,
):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    if algorithm == "rbc":
        if datatype == np.float64 or out_dtype in ["int32", np.int32]:
            pytest.skip("RBC does not support float64 dtype or int32 labels")
    if nrows == 500000 and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            nrows = nrows * max_gpu_memory // 32
        else:
            pytest.skip(
                "Insufficient GPU memory for this test. "
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    n_samples = nrows
    n_feats = ncols
    X, y = make_blobs(
        n_samples=n_samples,
        cluster_std=0.01,
        n_features=n_feats,
        random_state=0,
    )
    X = X.astype(datatype)

    handle, stream = get_handle(use_handle)

    eps = 1
    cuml_dbscan = cuDBSCAN(
        handle=handle,
        eps=eps,
        min_samples=2,
        algorithm=algorithm,
        max_mbytes_per_batch=max_mbytes_per_batch,
        output_type="numpy",
    )

    cu_labels = cuml_dbscan.fit_predict(X, out_dtype=out_dtype)

    if nrows < 500000:
        sk_dbscan = skDBSCAN(eps=1, min_samples=2, algorithm="brute")
        sk_labels = sk_dbscan.fit_predict(X)

        # Check the core points are equal
        assert array_equal(
            cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
        )

        # Check the labels are correct
        assert_dbscan_equal(
            sk_labels, cu_labels, X, cuml_dbscan.core_sample_indices_, eps
        )

    if out_dtype == "int32" or out_dtype == np.int32:
        assert cu_labels.dtype == np.int32
    elif out_dtype == "int64" or out_dtype == np.int64:
        assert cu_labels.dtype == np.int64


@pytest.mark.parametrize(
    "max_mbytes_per_batch",
    [unit_param(1), quality_param(1e2), stress_param(None)],
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(10000)]
)
@pytest.mark.parametrize("out_dtype", ["int32", "int64"])
def test_dbscan_precomputed(datatype, nrows, max_mbytes_per_batch, out_dtype):
    # 2-dimensional dataset for easy distance matrix computation
    X, y = make_blobs(
        n_samples=nrows, cluster_std=0.01, n_features=2, random_state=0
    )

    # Precompute distances
    X_dist = pairwise_distances(X).astype(datatype)

    eps = 1
    cuml_dbscan = cuDBSCAN(
        eps=eps,
        min_samples=2,
        metric="precomputed",
        max_mbytes_per_batch=max_mbytes_per_batch,
        output_type="numpy",
    )

    cu_labels = cuml_dbscan.fit_predict(X_dist, out_dtype=out_dtype)

    sk_dbscan = skDBSCAN(
        eps=eps, min_samples=2, metric="precomputed", algorithm="brute"
    )
    sk_labels = sk_dbscan.fit_predict(X_dist)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels, cu_labels, X, cuml_dbscan.core_sample_indices_, eps
    )


@pytest.mark.parametrize(
    "max_mbytes_per_batch",
    [unit_param(1), quality_param(1e2), stress_param(None)],
)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(10000)]
)
@pytest.mark.parametrize("out_dtype", ["int32", "int64"])
def test_dbscan_cosine(nrows, max_mbytes_per_batch, out_dtype):
    # 2-dimensional dataset for easy distance matrix computation
    X, y = make_blobs(
        n_samples=nrows, cluster_std=0.01, n_features=2, random_state=0
    )

    eps = 0.1

    cuml_dbscan = cuDBSCAN(
        eps=eps,
        min_samples=5,
        metric="cosine",
        max_mbytes_per_batch=max_mbytes_per_batch,
        output_type="numpy",
    )

    cu_labels = cuml_dbscan.fit_predict(X, out_dtype=out_dtype)

    sk_dbscan = skDBSCAN(
        eps=eps, min_samples=5, metric="cosine", algorithm="brute"
    )

    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels, cu_labels, X, cuml_dbscan.core_sample_indices_, eps
    )


@pytest.mark.parametrize("name", ["noisy_moons", "blobs", "no_structure"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
# Vary the eps to get a range of core point counts
@pytest.mark.parametrize("eps", [0.05, 0.1, 0.5])
def test_dbscan_sklearn_comparison(name, nrows, eps):
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    if nrows == 500000 and name == "blobs" and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            nrows = nrows * max_gpu_memory // 32
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    default_base = {
        "quantile": 0.2,
        "eps": eps,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 2,
    }
    n_samples = nrows
    pat = make_pattern(name, n_samples)
    params = default_base.copy()
    params.update(pat[1])
    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cuml_dbscan = cuDBSCAN(eps=eps, min_samples=5, output_type="numpy")
    cu_labels = cuml_dbscan.fit_predict(X)

    if nrows < 500000:
        sk_dbscan = skDBSCAN(eps=eps, min_samples=5)
        sk_labels = sk_dbscan.fit_predict(X)

        # Check the core points are equal
        assert array_equal(
            cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
        )

        # Check the labels are correct
        assert_dbscan_equal(
            sk_labels, cu_labels, X, cuml_dbscan.core_sample_indices_, eps
        )


@pytest.mark.parametrize("name", ["noisy_moons", "blobs", "no_structure"])
def test_dbscan_default(name):
    default_base = {
        "quantile": 0.3,
        "eps": 0.5,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 2,
    }
    n_samples = 500
    pat = make_pattern(name, n_samples)
    params = default_base.copy()
    params.update(pat[1])
    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cuml_dbscan = cuDBSCAN(output_type="numpy")
    cu_labels = cuml_dbscan.fit_predict(X)

    sk_dbscan = skDBSCAN(eps=params["eps"], min_samples=5)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels,
        cu_labels,
        X,
        cuml_dbscan.core_sample_indices_,
        params["eps"],
    )


@pytest.mark.xfail(strict=True, raises=ValueError)
def test_dbscan_out_dtype_fails_invalid_input():
    X, _ = make_blobs(n_samples=500)

    cuml_dbscan = cuDBSCAN(output_type="numpy")
    cuml_dbscan.fit_predict(X, out_dtype="bad_input")


def test_core_point_prop1():
    params = {"eps": 1.1, "min_samples": 4}

    # The input looks like a latin cross or a star with a chain:
    #   .
    # . . . . .
    #   .
    # There is 1 core-point (intersection of the bars)
    # and the two points to the very right are not reachable from it
    # So there should be one cluster (the plus/star on the left)
    # and two noise points

    X = np.array(
        [[0, 0], [1, 0], [1, 1], [1, -1], [2, 0], [3, 0], [4, 0]],
        dtype=np.float32,
    )
    cuml_dbscan = cuDBSCAN(**params)
    cu_labels = cuml_dbscan.fit_predict(X)

    sk_dbscan = skDBSCAN(**params)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )
    assert array_equal(cuml_dbscan.components_, sk_dbscan.components_)

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels,
        cu_labels,
        X,
        cuml_dbscan.core_sample_indices_,
        params["eps"],
    )


def test_core_point_prop2():
    params = {"eps": 1.1, "min_samples": 4}

    # The input looks like a long two-barred (orhodox) cross or
    # two stars next to each other:
    #   .     .
    # . . . . . .
    #   .     .
    # There are 2 core-points but they are not reachable from each other
    # So there should be two clusters, both in the form of a plus/star

    X = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, -1],
            [2, 0],
            [3, 0],
            [4, 0],
            [4, 1],
            [4, -1],
            [5, 0],
        ],
        dtype=np.float32,
    )
    cuml_dbscan = cuDBSCAN(**params)
    cu_labels = cuml_dbscan.fit_predict(X)

    sk_dbscan = skDBSCAN(**params)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )
    assert array_equal(cuml_dbscan.components_, sk_dbscan.components_)

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels,
        cu_labels,
        X,
        cuml_dbscan.core_sample_indices_,
        params["eps"],
    )


def test_core_point_prop3():
    params = {"eps": 1.1, "min_samples": 4}

    # The input looks like a two-barred (orhodox) cross or
    # two stars sharing a link:
    #   .   .
    # . . . . .
    #   .   .
    # There are 2 core-points but they are not reachable from each other
    # So there should be two clusters.
    # However, the link that is shared between the stars
    # actually has an ambiguous label (to the best of my knowledge)
    # as it will depend on the order in which we process the core-points.
    # So we exclude that point from the comparison with sklearn

    # TODO: the above text does not correspond to the actual test!

    X = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, -1],
            [3, 0],
            [4, 0],
            [4, 1],
            [4, -1],
            [5, 0],
            [2, 0],
        ],
        dtype=np.float32,
    )
    cuml_dbscan = cuDBSCAN(**params)
    cu_labels = cuml_dbscan.fit_predict(X)

    sk_dbscan = skDBSCAN(**params)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )
    assert array_equal(cuml_dbscan.components_, sk_dbscan.components_)

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels,
        cu_labels,
        X,
        cuml_dbscan.core_sample_indices_,
        params["eps"],
    )


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("out_dtype", ["int32", np.int32, "int64", np.int64])
@pytest.mark.parametrize("algorithm", ["brute", "rbc"])
@pytest.mark.parametrize("n_samples", [unit_param(500), stress_param(5000)])
def test_dbscan_propagation(
    datatype, use_handle, out_dtype, algorithm, n_samples
):
    if algorithm == "rbc":
        if datatype == np.float64 or out_dtype in ["int32", np.int32]:
            pytest.skip("RBC does not support float64 dtype or int32 labels")

    X, y = make_blobs(
        n_samples,
        centers=1,
        cluster_std=8.0,
        center_box=(-100.0, 100.0),
        random_state=8,
    )
    X = X.astype(datatype)

    handle, stream = get_handle(use_handle)
    eps = 0.5
    cuml_dbscan = cuDBSCAN(
        handle=handle,
        eps=eps,
        min_samples=5,
        algorithm=algorithm,
        output_type="numpy",
    )
    cu_labels = cuml_dbscan.fit_predict(X, out_dtype=out_dtype)

    sk_dbscan = skDBSCAN(eps=eps, min_samples=5)
    sk_labels = sk_dbscan.fit_predict(X)

    # Check the core points are equal
    assert array_equal(
        cuml_dbscan.core_sample_indices_, sk_dbscan.core_sample_indices_
    )

    # Check the labels are correct
    assert_dbscan_equal(
        sk_labels, cu_labels, X, cuml_dbscan.core_sample_indices_, eps
    )


def test_dbscan_no_calc_core_point_indices():

    params = {"eps": 1.1, "min_samples": 4}
    n_samples = 1000
    pat = make_pattern("noisy_moons", n_samples)

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    # Set calc_core_sample_indices=False
    cuml_dbscan = cuDBSCAN(
        eps=params["eps"],
        min_samples=5,
        output_type="numpy",
        calc_core_sample_indices=False,
    )
    cuml_dbscan.fit_predict(X)

    # Make sure we are None
    assert cuml_dbscan.core_sample_indices_ is None
    assert cuml_dbscan.components_ is None


def test_dbscan_on_empty_array():

    X = np.array([])
    cuml_dbscan = cuDBSCAN()

    with pytest.raises(ValueError):
        cuml_dbscan.fit(X)
