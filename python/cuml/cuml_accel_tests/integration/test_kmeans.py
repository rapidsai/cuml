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
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


@pytest.fixture(scope="module")
def clustering_data():
    X, y = make_blobs(
        n_samples=300, centers=3, cluster_std=1.0, random_state=42
    )
    return X, y


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


def test_kmeans_init_parameter():
    # Check that not passing a value for a constructor argument and passing the
    # scikit-learn default value leads to the same behavior.
    X, y = make_blobs(
        n_samples=300, centers=3, cluster_std=1.0, random_state=42
    )
    km1 = KMeans(init="k-means++")
    km1.fit(X, y)
    # Check that the translation of "k-means++" worked.
    assert km1.init == "scalable-k-means++"

    km2 = KMeans()
    km2.fit(X, y)
    # No init parameter should lead to the cuml default being used.
    assert km2.init == "scalable-k-means++"


def test_kmeans_sparse_cpu_dispatch():
    """Test that sparse inputs are dispatched to CPU in accel mode"""
    # Generate dense data
    X, y = make_blobs(
        n_samples=100,
        n_features=10,
        centers=3,
        cluster_std=1.0,
        random_state=42,
    )

    # Convert to sparse matrix
    X_sparse = csr_matrix(X)

    # Create KMeans instance
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Fit with sparse input
    kmeans.fit(X_sparse)

    # Verify that the model was fitted on CPU
    # This can be checked by verifying the model's attributes are numpy arrays
    assert hasattr(kmeans, "_cpu_model")
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert isinstance(kmeans.labels_, np.ndarray)

    # Verify predictions work with sparse input
    preds = kmeans.predict(X_sparse)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == X_sparse.shape[0]
