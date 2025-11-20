# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import pytest
from sklearn import cluster

from cuml.cluster import AgglomerativeClustering
from cuml.datasets import make_blobs
from cuml.metrics import adjusted_rand_score


@pytest.mark.parametrize("connectivity", ["knn", "pairwise"])
def test_duplicate_distances(connectivity):
    X = cp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])

    cuml_agg = AgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        linkage="single",
        connectivity=connectivity,
    )

    sk_agg = cluster.AgglomerativeClustering(
        n_clusters=2, metric="euclidean", linkage="single"
    )

    cuml_agg.fit(X)
    sk_agg.fit(X.get())

    assert adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0


@pytest.mark.parametrize("nrows", [100, 1000])
@pytest.mark.parametrize("ncols", [25, 50])
@pytest.mark.parametrize("nclusters", [1, 2, 10, 50])
@pytest.mark.parametrize("c", [3, 5, 15])
@pytest.mark.parametrize("connectivity", ["knn", "pairwise"])
def test_single_linkage_sklearn_compare(
    nrows, ncols, nclusters, c, connectivity
):
    X, y = make_blobs(
        int(nrows), ncols, nclusters, cluster_std=1.0, shuffle=False
    )

    cuml_agg = AgglomerativeClustering(
        n_clusters=nclusters,
        metric="euclidean",
        linkage="single",
        c=c,
        connectivity=connectivity,
    )

    cuml_agg.fit(X)

    sk_agg = cluster.AgglomerativeClustering(
        n_clusters=nclusters, metric="euclidean", linkage="single"
    )
    sk_agg.fit(cp.asnumpy(X))

    # Cluster assignments should be exact, even though the actual
    # labels may differ
    assert adjusted_rand_score(cuml_agg.labels_, sk_agg.labels_) == 1.0
    assert cuml_agg.n_connected_components_ == sk_agg.n_connected_components_
    assert cuml_agg.n_leaves_ == sk_agg.n_leaves_
    assert cuml_agg.n_clusters_ == sk_agg.n_clusters_
    # The children in the tree may differ, just compare shapes
    assert cuml_agg.children_.shape == sk_agg.children_.shape


def test_metric_none_deprecated():
    X, _ = make_blobs()

    model = AgglomerativeClustering(metric=None)
    with pytest.warns(FutureWarning, match="metric=None"):
        model.fit(X)

    assert hasattr(model, "labels_")


def test_n_neighobrs_deprecated():
    X, _ = make_blobs()

    model = AgglomerativeClustering(n_neighbors=15)
    with pytest.warns(FutureWarning, match="n_neighbors"):
        model.fit(X)

    assert hasattr(model, "labels_")


def test_invalid_inputs():
    X, _ = make_blobs()

    with pytest.raises(ValueError):
        AgglomerativeClustering(metric="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(connectivity="doesntexist").fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=0).fit(X)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=500).fit(X)
