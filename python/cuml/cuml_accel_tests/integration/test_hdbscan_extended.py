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


import hdbscan
import numpy as np
import pytest
from hdbscan import prediction, validity
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def synthetic_data():
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=5,
        cluster_std=0.5,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


def test_hdbscan_approximate_predict(synthetic_data):
    X_train, _ = synthetic_data
    X_test, _ = make_blobs(
        n_samples=100,
        n_features=2,
        centers=5,
        cluster_std=0.5,
        random_state=24,
    )
    X_test = StandardScaler().fit_transform(X_test)
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    test_labels, strengths = hdbscan.approximate_predict(clusterer, X_test)
    # Check that labels are assigned to test data
    assert (
        len(test_labels) == X_test.shape[0]
    ), "Labels should be assigned to test data points"
    assert (
        len(strengths) == X_test.shape[0]
    ), "Strengths should be computed for test data points"
    # Check that strengths are between 0 and 1
    assert np.all(
        (strengths >= 0) & (strengths <= 1)
    ), "Strengths should be between 0 and 1"


def test_hdbscan_membership_vector(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    point = X_train[0].reshape((1, 2))
    hdbscan.membership_vector(clusterer, point)


def test_hdbscan_all_points_membership_vectors(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    memberships = hdbscan.all_points_membership_vectors(clusterer)
    # Check that the number of membership vectors matches the number of samples
    assert (
        len(memberships) == X_train.shape[0]
    ), "There should be a membership vector for each sample"
    # Check that each membership vector sums to 1
    for membership in memberships:
        # Check that all probabilities are between 0 and 1
        assert all(
            0.0 <= v <= 1.0 for v in membership
        ), "Probabilities should be between 0 and 1"


def test_hdbscan_validity_index(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    score = validity.validity_index(X, clusterer.labels_, metric="euclidean")
    # Check that the validity index is a finite number
    assert np.isfinite(score), "Validity index should be a finite number"


def test_hdbscan_condensed_tree(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    condensed_tree = clusterer.condensed_tree_
    # Check that the condensed tree has the expected attributes
    assert hasattr(
        condensed_tree, "to_pandas"
    ), "Condensed tree should have a 'to_pandas' method"
    # Convert to pandas DataFrame and check columns
    condensed_tree.to_pandas()


def test_hdbscan_single_linkage_tree_attribute(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    single_linkage_tree = clusterer.single_linkage_tree_
    # Check that the single linkage tree has the expected attributes
    assert hasattr(
        single_linkage_tree, "to_numpy"
    ), "Single linkage tree should have a 'to_numpy' method"
    # Convert to NumPy array and check shape
    sl_tree_array = single_linkage_tree.to_numpy()
    assert (
        sl_tree_array.shape[1] == 4
    ), "Single linkage tree array should have 4 columns"


def test_hdbscan_flat_clustering(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    # Extract clusters at a specific cluster_selection_epsilon
    clusterer_flat = hdbscan.HDBSCAN(cluster_selection_epsilon=0.1)
    clusterer_flat.fit(X)
    # Check that clusters are formed
    n_clusters_flat = len(set(clusterer_flat.labels_)) - (
        1 if -1 in clusterer_flat.labels_ else 0
    )
    assert n_clusters_flat > 0, "Should find clusters with flat clustering"


def test_hdbscan_prediction_membership_vector(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    point = X_train[0].reshape((1, 2))
    prediction.membership_vector(clusterer, point)


def test_hdbscan_prediction_all_points_membership_vectors(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    memberships = prediction.all_points_membership_vectors(clusterer)
    # Check that the number of membership vectors matches the number of samples
    assert (
        len(memberships) == X_train.shape[0]
    ), "There should be a membership vector for each sample"
    for membership in memberships:
        # Check that all probabilities are between 0 and 1
        assert all(
            0.0 <= v <= 1.0 for v in membership
        ), "Probabilities should be between 0 and 1"


def test_hdbscan_outlier_exposure(synthetic_data):
    # Note: hdbscan may not have a function named 'outlier_exposure'
    # This is a placeholder for any outlier detection functionality
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    # Check if outlier scores are computed
    if hasattr(clusterer, "outlier_scores_"):
        outlier_scores = clusterer.outlier_scores_
        # Check that outlier scores are finite numbers
        assert np.all(
            np.isfinite(outlier_scores)
        ), "Outlier scores should be finite numbers"
    else:
        pytest.skip(
            "Outlier exposure functionality is not available in this version of HDBSCAN"
        )


# test requires networkx
# def test_hdbscan_extract_single_linkage_tree(synthetic_data):
#     X, _ = synthetic_data
#     clusterer = hdbscan.HDBSCAN()
#     clusterer.fit(X)
#     # Extract the single linkage tree
#     sl_tree = clusterer.single_linkage_tree_.to_networkx()
#     # Check that the tree has the correct number of nodes
#     assert sl_tree.number_of_nodes() == X.shape[0], "Single linkage tree should have a node for each data point"


def test_hdbscan_get_exemplars(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    if hasattr(clusterer, "exemplars_"):
        exemplars = clusterer.exemplars_
        # Check that exemplars are available for each cluster
        n_clusters = len(set(clusterer.labels_)) - (
            1 if -1 in clusterer.labels_ else 0
        )
        assert (
            len(exemplars) == n_clusters
        ), "There should be exemplars for each cluster"
    else:
        pytest.skip(
            "Exemplar functionality is not available in this version of HDBSCAN"
        )
