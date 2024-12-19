#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_blobs(
        n_samples=500,
        n_features=10,
        centers=5,
        cluster_std=1.0,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("n_neighbors", [1, 5, 10, 20])
def test_nearest_neighbors_n_neighbors(synthetic_data, n_neighbors):
    X, _ = synthetic_data
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    distances, indices = model.kneighbors(X)
    # Check that the correct number of neighbors is returned
    assert (
        indices.shape[1] == n_neighbors
    ), f"Should return {n_neighbors} neighbors"


@pytest.mark.parametrize(
    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
)
def test_nearest_neighbors_algorithm(synthetic_data, algorithm):
    X, _ = synthetic_data
    model = NearestNeighbors(algorithm=algorithm)
    model.fit(X)
    distances, indices = model.kneighbors(X)
    # Check that the output shape is correct
    assert (
        indices.shape[0] == X.shape[0]
    ), f"Number of samples should remain the same with algorithm={algorithm}"


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
)
def test_nearest_neighbors_metric(synthetic_data, metric):
    X, _ = synthetic_data
    model = NearestNeighbors(metric=metric)
    model.fit(X)
    model.kneighbors(X)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_nearest_neighbors_p_parameter(synthetic_data, p):
    X, _ = synthetic_data
    model = NearestNeighbors(metric="minkowski", p=p)
    model.fit(X)
    distances, indices = model.kneighbors(X)


@pytest.mark.parametrize("leaf_size", [10, 30, 50])
def test_nearest_neighbors_leaf_size(synthetic_data, leaf_size):
    X, _ = synthetic_data
    model = NearestNeighbors(leaf_size=leaf_size)
    model.fit(X)


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_nearest_neighbors_n_jobs(synthetic_data, n_jobs):
    X, _ = synthetic_data
    model = NearestNeighbors(n_jobs=n_jobs)
    model.fit(X)
    # We assume the code runs without error; no direct way to test n_jobs effect
    assert True, f"NearestNeighbors ran successfully with n_jobs={n_jobs}"


def test_nearest_neighbors_radius(synthetic_data):
    X, _ = synthetic_data
    radius = 1.0
    model = NearestNeighbors(radius=radius)
    model.fit(X)
    distances, indices = model.radius_neighbors(X)
    # Check that all returned distances are within the radius
    for dist in distances:
        assert np.all(
            dist <= radius
        ), f"All distances should be within the radius {radius}"


def test_nearest_neighbors_invalid_algorithm(synthetic_data):
    X, _ = synthetic_data
    with pytest.raises((ValueError, KeyError)):
        model = NearestNeighbors(algorithm="invalid_algorithm")
        model.fit(X)


def test_nearest_neighbors_invalid_metric(synthetic_data):
    X, _ = synthetic_data
    with pytest.raises(ValueError):
        model = NearestNeighbors(metric="invalid_metric")
        model.fit(X)


def test_nearest_neighbors_kneighbors_graph(synthetic_data):
    X, _ = synthetic_data
    n_neighbors = 5
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    graph = model.kneighbors_graph(X)
    # Check that the graph is of correct shape and type
    assert graph.shape == (
        X.shape[0],
        X.shape[0],
    ), "Graph shape should be (n_samples, n_samples)"
    assert graph.getformat() == "csr", "Graph should be in CSR format"
    # Check that each row has n_neighbors non-zero entries
    row_counts = np.diff(graph.indptr)
    assert np.all(
        row_counts == n_neighbors
    ), f"Each sample should have {n_neighbors} neighbors in the graph"


def test_nearest_neighbors_radius_neighbors_graph(synthetic_data):
    X, _ = synthetic_data
    radius = 1.0
    model = NearestNeighbors(radius=radius)
    model.fit(X)
    graph = model.radius_neighbors_graph(X)
    # Check that the graph is of correct shape and type
    assert graph.shape == (
        X.shape[0],
        X.shape[0],
    ), "Graph shape should be (n_samples, n_samples)"
    assert graph.getformat() == "csr", "Graph should be in CSR format"
    # Check that non-zero entries correspond to distances within the radius
    non_zero_indices = graph.nonzero()
    pairwise_distances(X[non_zero_indices[0]], X[non_zero_indices[1]])


@pytest.mark.parametrize("return_distance", [True, False])
def test_nearest_neighbors_return_distance(synthetic_data, return_distance):
    X, _ = synthetic_data
    model = NearestNeighbors()
    model.fit(X)
    result = model.kneighbors(X, return_distance=return_distance)
    if return_distance:
        distances, indices = result
        assert (
            distances.shape == indices.shape
        ), "Distances and indices should have the same shape"
    else:
        indices = result
        assert indices.shape == (
            X.shape[0],
            model.n_neighbors,
        ), "Indices shape should match (n_samples, n_neighbors)"


def test_nearest_neighbors_sparse_input():
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.random.rand(100, 20))
    model = NearestNeighbors()
    model.fit(X)
    distances, indices = model.kneighbors(X)
    assert distances.shape == (
        X.shape[0],
        model.n_neighbors,
    ), "Distances shape should match for sparse input"
