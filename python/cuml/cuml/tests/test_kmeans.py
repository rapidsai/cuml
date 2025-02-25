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

from cuml.internals.safe_imports import gpu_only_import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from cuml.testing.utils import (
    get_pattern,
    unit_param,
    quality_param,
    stress_param,
    array_equal,
)
from cuml.datasets import make_blobs
import pytest
import random

import cuml
import cuml.internals.logger as logger
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


cp = gpu_only_import("cupy")


dataset_names = ["blobs", "noisy_circles", "noisy_moons", "varied", "aniso"]


@pytest.fixture
def get_data_consistency_test():
    cluster_std = 1.0
    nrows = 1000
    ncols = 50
    nclusters = 8

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )
    return X, y


@pytest.fixture
def random_state():
    random_state = random.randint(0, 10**6)
    with logger.set_level(logger.level_enum.debug):
        logger.debug("Random seed: {}".format(random_state))
    return random_state


def test_n_init_deprecation():
    X, y = make_blobs(
        random_state=0,
    )

    # Warn about default changing
    kmeans = cuml.KMeans()
    with pytest.warns(
        FutureWarning, match="The default value of `n_init` will change from"
    ):
        kmeans.fit(X)

    # No warning when explicitly set to integer or 'auto'
    for n_init in ("auto", 2):
        kmeans = cuml.KMeans(n_init=n_init)
        kmeans.fit(X)


@pytest.mark.xfail
def test_n_init_cluster_consistency(random_state):

    nclusters = 8
    X, y = get_data_consistency_test()

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)
    initial_clusters = cuml_kmeans.cluster_centers_

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)

    assert array_equal(initial_clusters, cuml_kmeans.cluster_centers_)


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [2, 5])
def test_traditional_kmeans_plus_plus_init(
    nrows, ncols, nclusters, random_state
):

    # Using fairly high variance between points in clusters
    cluster_std = 1.0

    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X)
    cu_score = cuml_kmeans.score(X)

    kmeans = cluster.KMeans(
        random_state=random_state, n_clusters=nclusters, n_init=1
    )
    kmeans.fit(cp.asnumpy(X))
    sk_score = kmeans.score(cp.asnumpy(X))

    cp.testing.assert_allclose(cu_score, sk_score, atol=0.1, rtol=1e-4)


@pytest.mark.parametrize("nrows", [100, 500])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [5, 10])
@pytest.mark.parametrize("max_weight", [10])
def test_weighted_kmeans(nrows, ncols, nclusters, max_weight, random_state):

    # Using fairly high variance between points in clusters
    cluster_std = 1.0
    np.random.seed(random_state)

    # set weight per sample to be from 1 to max_weight
    wt = np.random.randint(1, high=max_weight, size=nrows)

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_kmeans.fit(X, sample_weight=wt)
    cu_score = cuml_kmeans.score(X)

    sk_kmeans = cluster.KMeans(
        random_state=random_state, n_clusters=nclusters, n_init=1
    )
    sk_kmeans.fit(cp.asnumpy(X), sample_weight=wt)
    sk_score = sk_kmeans.score(cp.asnumpy(X))

    if cu_score < sk_score:
        relative_tolerance = 0.1  # allow 10% difference
        diff = abs(cu_score - sk_score)
        avg_score = (abs(cu_score) + abs(sk_score)) / 2.0
        assert diff / avg_score <= relative_tolerance


@pytest.mark.parametrize("nrows", [1000, 10000])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [2, 5])
@pytest.mark.parametrize("cluster_std", [1.0, 0.1, 0.01])
def test_kmeans_clusters_blobs(
    nrows, ncols, nclusters, random_state, cluster_std
):

    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=random_state,
        output_type="numpy",
        n_init=1,
    )

    preds = cuml_kmeans.fit_predict(X)

    assert adjusted_rand_score(cp.asnumpy(preds), cp.asnumpy(y)) >= 0.99


@pytest.mark.parametrize("name", dataset_names)
@pytest.mark.parametrize("nrows", [unit_param(1000), quality_param(5000)])
def test_kmeans_sklearn_comparison(name, nrows, random_state):

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 3,
    }

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(
        n_clusters=params["n_clusters"],
        output_type="numpy",
        init="k-means++",
        random_state=random_state,
        n_init=10,
    )

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(
        random_state=random_state,
        n_clusters=params["n_clusters"],
        n_init=10,
    )
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize("name", dataset_names)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
def test_kmeans_sklearn_comparison_default(name, nrows, random_state):

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 3,
    }

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(
        n_clusters=params["n_clusters"],
        random_state=random_state,
        n_init=10,
        output_type="numpy",
    )

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)
    cu_score = adjusted_rand_score(cu_y_pred, y)
    kmeans = cluster.KMeans(
        random_state=random_state,
        n_clusters=params["n_clusters"],
        n_init=10,
    )
    sk_y_pred = kmeans.fit_predict(X)
    sk_score = adjusted_rand_score(sk_y_pred, y)

    assert sk_score - 1e-2 <= cu_score <= sk_score + 1e-2


@pytest.mark.parametrize(
    "max_iter, oversampling_factor, max_samples_per_batch, init",
    [
        (100, 0.5, 1 << 10, "preset"),
        (1000, 1.0, 1 << 15, "preset"),
        (500, 1.5, 1 << 5, "k-means||"),
        (1000, 1.0, 1 << 10, "random"),
        # Redundant case to better exercise 'k-means||'
        (1000, 1.0, 1 << 15, "k-means||"),
    ],
)
@pytest.mark.parametrize(
    "n_clusters", [unit_param(10), unit_param(100), stress_param(1000)]
)
def test_all_kmeans_params(
    n_clusters,
    max_iter,
    init,
    oversampling_factor,
    max_samples_per_batch,
    random_state,
):

    np.random.seed(0)
    X = np.random.rand(1000, 10)

    if init == "preset":
        init = np.random.rand(n_clusters, 10)

    cuml_kmeans = cuml.KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init=init,
        random_state=random_state,
        oversampling_factor=oversampling_factor,
        max_samples_per_batch=max_samples_per_batch,
        output_type="cupy",
        n_init=1,
    )

    cuml_kmeans.fit_predict(X)


@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
def test_score(nrows, ncols, nclusters, random_state):

    X, y = make_blobs(
        int(nrows),
        ncols,
        nclusters,
        cluster_std=1.0,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=random_state,
        output_type="numpy",
        n_init=1,
    )

    cuml_kmeans.fit(X)

    actual_score = cuml_kmeans.score(X)
    predictions = cuml_kmeans.predict(X)

    centers = cuml_kmeans.cluster_centers_

    expected_score = 0.0
    for idx, label in enumerate(predictions):
        x = X[idx, :]
        y = cp.array(centers[label, :], dtype=cp.float32)

        sq_euc_dist = cp.sum(cp.square((x - y)))
        expected_score += sq_euc_dist

    expected_score *= -1

    cp.testing.assert_allclose(
        actual_score, expected_score, atol=0.1, rtol=1e-4
    )


@pytest.mark.parametrize("nrows", [100])
@pytest.mark.parametrize("ncols", [25])
@pytest.mark.parametrize("nclusters", [5])
@pytest.mark.parametrize("max_weight", [10])
def test_fit_transform_weighted_kmeans(
    nrows, ncols, nclusters, max_weight, random_state
):

    # Using fairly high variance between points in clusters
    cluster_std = 1.0
    np.random.seed(random_state)

    # set weight per sample to be from 1 to max_weight
    wt = np.random.randint(1, high=max_weight, size=nrows)

    X, y = make_blobs(
        nrows,
        ncols,
        nclusters,
        cluster_std=cluster_std,
        shuffle=False,
        random_state=0,
    )

    cuml_kmeans = cuml.KMeans(
        init="k-means++",
        n_clusters=nclusters,
        n_init=10,
        random_state=random_state,
        output_type="numpy",
    )

    cuml_transf = cuml_kmeans.fit_transform(X, sample_weight=wt)
    cu_score = cuml_kmeans.score(X)

    sk_kmeans = cluster.KMeans(random_state=random_state, n_clusters=nclusters)
    sk_transf = sk_kmeans.fit_transform(cp.asnumpy(X), sample_weight=wt)
    sk_score = sk_kmeans.score(cp.asnumpy(X))

    # we fail if cuML's score is significantly worse than sklearn's
    if cu_score < sk_score:
        relative_tolerance = 0.1  # allow 10% difference
        diff = abs(cu_score - sk_score)
        avg_score = (abs(cu_score) + abs(sk_score)) / 2.0
        assert diff / avg_score <= relative_tolerance

    assert sk_transf.shape == cuml_transf.shape
