#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def clustering_data():
    X, y = make_blobs(
        n_samples=300, centers=3, cluster_std=1.0, random_state=42
    )
    return X.astype(np.float32), y


def test_spectral_clustering_default(clustering_data):
    X, y = clustering_data
    sc = SpectralClustering(affinity="nearest_neighbors", random_state=42).fit(
        X
    )
    assert sc.labels_.shape == y.shape


@pytest.mark.parametrize("n_clusters", [2, 3, 4, 5])
def test_spectral_clustering_n_clusters(clustering_data, n_clusters):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("n_neighbors", [5, 10, 20])
def test_spectral_clustering_n_neighbors(clustering_data, n_neighbors):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("n_components", [2, 3, 5])
def test_spectral_clustering_n_components(clustering_data, n_components):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        n_components=n_components,
        affinity="nearest_neighbors",
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("n_init", [1, 5, 10])
def test_spectral_clustering_n_init(clustering_data, n_init):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        n_init=n_init,
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("eigen_tol", ["auto", 0.0, 1e-4])
def test_spectral_clustering_eigen_tol(clustering_data, eigen_tol):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        eigen_tol=eigen_tol,
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize(
    "assign_labels", ["kmeans", "discretize", "cluster_qr"]
)
def test_spectral_clustering_assign_labels(clustering_data, assign_labels):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        assign_labels=assign_labels,
        random_state=42,
    ).fit(X)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


def test_spectral_clustering_precomputed(clustering_data):
    from sklearn.neighbors import kneighbors_graph

    X, y_true = clustering_data
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=True)
    affinity_matrix = 0.5 * (connectivity + connectivity.T)
    sc = SpectralClustering(
        n_clusters=3,
        affinity="precomputed",
        random_state=42,
    ).fit(affinity_matrix)
    y_pred = sc.labels_
    adjusted_rand_score(y_true, y_pred)


def test_spectral_clustering_fit_predict(clustering_data):
    X, y_true = clustering_data
    sc = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        random_state=42,
    )
    labels = sc.fit_predict(X)
    assert labels.shape == y_true.shape
    assert np.array_equal(labels, sc.labels_)


def test_spectral_clustering_random_state(clustering_data):
    X, _ = clustering_data
    sc1 = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        random_state=42,
    ).fit(X)
    sc2 = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        random_state=42,
    ).fit(X)
    assert np.array_equal(sc1.labels_, sc2.labels_), (
        "Results should be consistent with the same random_state"
    )
