# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering as skSpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

from cuml.cluster import SpectralClustering


@pytest.mark.parametrize(
    "affinity",
    [
        "precomputed",
        "nearest_neighbors",
    ],
)
@pytest.mark.parametrize(
    "n_samples,n_clusters",
    [
        (1000, 3),
        (2000, 4),
        (3000, 5),
    ],
)
def test_spectral_clustering_adjusted_rand_score(
    affinity, n_samples, n_clusters
):
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )
    X = X.astype(np.float32)

    if affinity == "precomputed":
        knn_graph = kneighbors_graph(
            X,
            n_neighbors=10,
            mode="connectivity",
            include_self=True,
        )
        knn_graph = 0.5 * (knn_graph + knn_graph.T)

        sk_spectral = skSpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
        y_sklearn = sk_spectral.fit_predict(knn_graph)

        cuml_spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
        )
        y_cuml_gpu = cuml_spectral.fit_predict(knn_graph)
        y_cuml = cp.asnumpy(y_cuml_gpu)
    else:
        sk_spectral = skSpectralClustering(
            n_clusters=n_clusters,
            n_neighbors=10,
            affinity="nearest_neighbors",
            random_state=42,
        )
        y_sklearn = sk_spectral.fit_predict(X)

        X_gpu = cp.asarray(X)
        cuml_spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=42,
        )
        y_cuml_gpu = cuml_spectral.fit_predict(X_gpu)
        y_cuml = cp.asnumpy(y_cuml_gpu)

    ari_sklearn = adjusted_rand_score(y_true, y_sklearn)
    ari_cuml = adjusted_rand_score(y_true, y_cuml)

    min_ari = 0.8
    assert ari_sklearn > min_ari
    assert ari_cuml > min_ari


@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        ("numpy", np.ndarray),
        ("cupy", cp.ndarray),
    ],
)
def test_output_type_handling(input_type, expected_type):
    n_samples = 1000
    n_clusters = 3

    X_np, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )
    X_np = X_np.astype(np.float32)
    X = X_np if input_type == "numpy" else cp.asarray(X_np)

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=30,
        random_state=42,
    )
    model.fit(X)
    assert isinstance(model.labels_, expected_type)
    assert model.labels_.shape == (n_samples,)

    out = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=30,
        random_state=42,
    ).fit_predict(X)
    assert isinstance(out, expected_type)
    assert out.shape == (n_samples,)


@pytest.mark.parametrize(
    "converter",
    [
        pytest.param(lambda x: x.toarray(), id="numpy"),
        pytest.param(lambda x: cp.asarray(x.toarray()), id="cupy"),
        pytest.param(sp.coo_matrix, id="scipy_coo"),
        pytest.param(sp.csr_matrix, id="scipy_csr"),
        pytest.param(sp.csc_matrix, id="scipy_csc"),
        pytest.param(cp_sp.coo_matrix, id="cupy_coo"),
        pytest.param(cp_sp.csr_matrix, id="cupy_csr"),
        pytest.param(cp_sp.csc_matrix, id="cupy_csc"),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_precomputed_matrix_formats(converter, dtype):
    n_samples = 1000
    n_clusters = 3

    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=0.3,
        random_state=42,
    )

    knn_graph = kneighbors_graph(
        X,
        n_neighbors=30,
        mode="connectivity",
        include_self=True,
    )
    knn_graph = 0.5 * (knn_graph + knn_graph.T)

    affinity_matrix = converter(knn_graph).astype(dtype)

    model = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=42
    )
    labels = model.fit_predict(affinity_matrix)

    labels_np = (
        cp.asnumpy(labels)
        if hasattr(labels, "get") or isinstance(labels, cp.ndarray)
        else labels
    )

    ari = adjusted_rand_score(y_true, labels_np)

    assert labels_np.shape == (n_samples,)

    min_ari = 0.8
    assert ari > min_ari
