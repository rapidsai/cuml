#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def clustering_data():
    X, y = make_blobs(
        n_samples=300, centers=3, cluster_std=1.0, random_state=42
    )
    return X, y


def test_kmeans_default(clustering_data):
    X, y = clustering_data
    kmeans = KMeans().fit(X, y)
    assert kmeans.labels_.shape == y.shape


@pytest.mark.parametrize("n_clusters", [2, 3, 4, 5])
def test_kmeans_n_clusters(clustering_data, n_clusters):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_init(clustering_data, init):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=3, init=init, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("n_init", [1, 5, 10, 20])
def test_kmeans_n_init(clustering_data, n_init):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=3, n_init=n_init, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("max_iter", [100, 300, 500])
def test_kmeans_max_iter(clustering_data, max_iter):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=3, max_iter=max_iter, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("tol", [1e-4, 1e-3, 1e-2])
def test_kmeans_tol(clustering_data, tol):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=3, tol=tol, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("algorithm", ["elkan", "lloyd"])
def test_kmeans_algorithm(clustering_data, algorithm):
    X, y_true = clustering_data
    kmeans = KMeans(n_clusters=3, algorithm=algorithm, random_state=42).fit(X)
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


@pytest.mark.parametrize("copy_x", [True, False])
def test_kmeans_copy_x(clustering_data, copy_x):
    X, y_true = clustering_data
    X_original = X.copy()
    kmeans = KMeans(n_clusters=3, copy_x=copy_x, random_state=42).fit(X)
    if copy_x:
        # X should remain unchanged
        assert np.allclose(
            X, X_original
        ), "X has been modified when copy_x=True"
    else:
        # X might be modified when copy_x=False
        pass  # We cannot guarantee X remains unchanged
    y_pred = kmeans.labels_
    adjusted_rand_score(y_true, y_pred)


def test_kmeans_random_state(clustering_data):
    X, y_true = clustering_data
    kmeans1 = KMeans(n_clusters=3, random_state=42).fit(X)
    kmeans2 = KMeans(n_clusters=3, random_state=42).fit(X)
    # With the same random_state, results should be the same
    assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_)
