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
        n_neighbors=3,
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
@pytest.mark.parametrize("k", [3, 5, 15])
@pytest.mark.parametrize("connectivity", ["knn", "pairwise"])
def test_single_linkage_sklearn_compare(
    nrows, ncols, nclusters, k, connectivity
):

    X, y = make_blobs(
        int(nrows), ncols, nclusters, cluster_std=1.0, shuffle=False
    )

    cuml_agg = AgglomerativeClustering(
        n_clusters=nclusters,
        metric="euclidean",
        linkage="single",
        n_neighbors=k,
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


def test_invalid_inputs():

    # Test bad metric
    with pytest.raises(ValueError):
        AgglomerativeClustering(metric="doesntexist")

    with pytest.raises(ValueError):
        AgglomerativeClustering(linkage="doesntexist")

    with pytest.raises(ValueError):
        AgglomerativeClustering(connectivity="doesntexist")

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_neighbors=1)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_neighbors=1024)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=0)

    with pytest.raises(ValueError):
        AgglomerativeClustering(n_clusters=500).fit(cp.ones((2, 5)))
