# Copyright (c) 2022, NVIDIA CORPORATION.
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

from cuml.datasets import make_blobs
import numpy as np
import cupy as cp
import umap.distances as dist
from cuml.manifold.umap import UMAP
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
from cuml.manifold.simpl_set import fuzzy_simplicial_set \
    as cu_fuzzy_simplicial_set
from umap.umap_ import simplicial_set_embedding as ref_simplicial_set_embedding
from cuml.manifold.simpl_set import simplicial_set_embedding \
    as cu_simplicial_set_embedding


def correctness_dense(a, b, rtol=0.1, threshold=0.8):
    n_elms = a.size
    n_correct = (cp.abs(a - b) <= (rtol * cp.abs(b))).sum()
    correctness = n_correct / n_elms
    return correctness >= threshold


def correctness_sparse(a, b, atol=0.1, rtol=0.2, threshold=0.8):
    n_ref_zeros = (a == 0).sum()
    n_ref_non_zero_elms = a.size - n_ref_zeros
    n_correct = (cp.abs(a - b) <= (atol + rtol * cp.abs(b))).sum()
    correctness = (n_correct - n_ref_zeros) / n_ref_non_zero_elms
    return correctness >= threshold


def test_fuzzy_simplicial_set():
    n_rows = 10000
    n_features = 16
    n_clusters = 30
    n_neighbors = 5
    random_state = 42
    metric = 'euclidean'

    X, _ = make_blobs(n_samples=n_rows, centers=n_clusters,
                      n_features=n_features, random_state=random_state)
    X = X.get()

    ref_fss_graph = ref_fuzzy_simplicial_set(X,
                                             n_neighbors,
                                             random_state,
                                             metric)[0].tocoo()
    cu_fss_graph = cu_fuzzy_simplicial_set(X,
                                           n_neighbors,
                                           random_state,
                                           metric)

    ref_fss_graph = cp.sparse.coo_matrix(ref_fss_graph).todense()
    cu_fss_graph = cu_fss_graph.todense()
    assert correctness_sparse(ref_fss_graph,
                              cu_fss_graph,
                              atol=0.1,
                              rtol=0.2,
                              threshold=0.8)


def test_simplicial_set_embedding():
    n_rows = 10000
    n_features = 16
    n_clusters = 30
    n_neighbors = 5
    random_state = 42
    metric = 'euclidean'
    n_components = 3
    initial_alpha = 1.0
    a, b = UMAP.find_ab_params(1.0, 0.1)
    gamma = 0
    negative_sample_rate = 5
    n_epochs = 500
    init = 'random'
    metric = 'euclidean'
    metric_kwds = {}
    densmap = False
    densmap_kwds = {}
    output_dens = False
    output_metric = 'euclidean'
    output_metric_kwds = {}

    X, _ = make_blobs(n_samples=n_rows, centers=n_clusters,
                      n_features=n_features, random_state=random_state)
    X = X.get()

    ref_fss_graph = ref_fuzzy_simplicial_set(X,
                                             n_neighbors,
                                             random_state,
                                             metric)[0]
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
        output_metric_kwds=output_metric_kwds)[0]

    cu_fss_graph = cu_fuzzy_simplicial_set(X,
                                           n_neighbors,
                                           random_state,
                                           metric)

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
        output_metric_kwds=output_metric_kwds)

    ref_embedding = cp.array(ref_embedding)
    assert correctness_dense(ref_embedding,
                             cu_embedding,
                             rtol=0.1,
                             threshold=0.8)
