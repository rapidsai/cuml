# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import platform
from cuml.manifold.umap import (
    simplicial_set_embedding as cu_simplicial_set_embedding,
)
from cuml.manifold.umap import fuzzy_simplicial_set as cu_fuzzy_simplicial_set
from cuml.neighbors import NearestNeighbors
from cuml.manifold.umap import UMAP
from cuml.internals.safe_imports import gpu_only_import
import pytest
from cuml.datasets import make_blobs
from cuml.internals.safe_imports import cpu_only_import
from cuml.metrics import trustworthiness

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


IS_ARM = platform.processor() == "aarch64"

if not IS_ARM:
    from umap.umap_ import (
        simplicial_set_embedding as ref_simplicial_set_embedding,
    )
    from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
    import umap.distances as dist


def correctness_dense(a, b, rtol=0.1, threshold=0.95):
    n_elms = a.size
    n_correct = (cp.abs(a - b) <= (rtol * cp.abs(b))).sum()
    correctness = n_correct / n_elms
    return correctness >= threshold


def correctness_sparse(a, b, atol=0.1, rtol=0.2, threshold=0.95):
    n_ref_zeros = (a == 0).sum()
    n_ref_non_zero_elms = a.size - n_ref_zeros
    n_correct = (cp.abs(a - b) <= (atol + rtol * cp.abs(b))).sum()
    correctness = (n_correct - n_ref_zeros) / n_ref_non_zero_elms
    return correctness >= threshold


@pytest.mark.parametrize("n_rows", [800, 5000])
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.parametrize("precomputed_nearest_neighbors", [False, True])
@pytest.mark.skipif(
    IS_ARM, reason="https://github.com/rapidsai/cuml/issues/5441"
)
def test_fuzzy_simplicial_set(
    n_rows, n_features, n_neighbors, precomputed_nearest_neighbors
):
    n_clusters = 30
    random_state = 42
    metric = "euclidean"

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )

    if precomputed_nearest_neighbors:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X)
        knn_dists, knn_indices = nn.kneighbors(
            X, n_neighbors, return_distance=True
        )
        cu_fss_graph = cu_fuzzy_simplicial_set(
            X,
            n_neighbors,
            random_state,
            metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )

        knn_indices = knn_indices.get()
        knn_dists = knn_dists.get()
        ref_fss_graph = ref_fuzzy_simplicial_set(
            X,
            n_neighbors,
            random_state,
            metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )[0].tocoo()
    else:
        cu_fss_graph = cu_fuzzy_simplicial_set(
            X, n_neighbors, random_state, metric
        )

        X = X.get()
        ref_fss_graph = ref_fuzzy_simplicial_set(
            X, n_neighbors, random_state, metric
        )[0].tocoo()

    cu_fss_graph = cu_fss_graph.todense()
    ref_fss_graph = cupyx.scipy.sparse.coo_matrix(ref_fss_graph).todense()
    assert correctness_sparse(
        ref_fss_graph, cu_fss_graph, atol=0.1, rtol=0.2, threshold=0.95
    )


@pytest.mark.parametrize("n_rows", [800, 5000])
@pytest.mark.parametrize("n_features", [8, 32])
@pytest.mark.parametrize("n_neighbors", [8, 16])
@pytest.mark.parametrize("n_components", [2, 5])
@pytest.mark.skipif(
    IS_ARM, reason="https://github.com/rapidsai/cuml/issues/5441"
)
def test_simplicial_set_embedding(
    n_rows, n_features, n_neighbors, n_components
):
    n_clusters = 30
    random_state = 42
    metric = "euclidean"
    initial_alpha = 1.0
    a, b = UMAP.find_ab_params(1.0, 0.1)
    gamma = 1.0
    negative_sample_rate = 5
    n_epochs = 500
    init = "random"
    metric = "euclidean"
    metric_kwds = {}
    densmap = False
    densmap_kwds = {}
    output_dens = False
    output_metric = "euclidean"
    output_metric_kwds = {}

    X, _ = make_blobs(
        n_samples=n_rows,
        centers=n_clusters,
        n_features=n_features,
        random_state=random_state,
    )
    X = X.get()

    ref_fss_graph = ref_fuzzy_simplicial_set(
        X, n_neighbors, random_state, metric
    )[0]
    ref_embedding = ref_simplicial_set_embedding(
        X,
        ref_fss_graph,
        n_components,
        initial_alpha,
        a,
        b,
        gamma,
        negative_sample_rate,
        n_epochs,
        init,
        np.random.RandomState(random_state),
        dist.named_distances_with_gradients[metric],
        metric_kwds,
        densmap,
        densmap_kwds,
        output_dens,
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
    )[0]

    cu_fss_graph = cu_fuzzy_simplicial_set(
        X, n_neighbors, random_state, metric
    )
    cu_embedding = cu_simplicial_set_embedding(
        X,
        cu_fss_graph,
        n_components,
        initial_alpha,
        a,
        b,
        gamma,
        negative_sample_rate,
        n_epochs,
        init,
        random_state,
        metric,
        metric_kwds,
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
    )

    ref_t_score = trustworthiness(X, ref_embedding, n_neighbors=n_neighbors)
    t_score = trustworthiness(X, cu_embedding, n_neighbors=n_neighbors)
    abs_tol = 0.05
    assert t_score >= ref_t_score - abs_tol
