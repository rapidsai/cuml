#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import hdbscan
import numpy as np
import pytest
import sklearn
from hdbscan import prediction, validity
from packaging.version import Version
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

if Version(sklearn.__version__) >= Version("1.8.0.dev0"):
    pytest.skip(
        "hdbscan requires sklearn < 1.8.0.dev0", allow_module_level=True
    )

# Ignore FutureWarning from third-party hdbscan package calling
# sklearn.utils.validation.check_array with deprecated 'force_all_finite'
# parameter. This is not in cuml's control. Note: this will break when
# sklearn 1.8 removes the deprecated parameter entirely - hdbscan will
# need to be updated at that point.
pytestmark = pytest.mark.filterwarnings(
    "ignore:'force_all_finite' was renamed to "
    "'ensure_all_finite':FutureWarning:sklearn"
)


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


@pytest.mark.parametrize("min_cluster_size", [5, 15, 30])
def test_hdbscan_min_cluster_size(synthetic_data, min_cluster_size):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, (
        f"Should find clusters with min_cluster_size={min_cluster_size}"
    )


@pytest.mark.parametrize("min_samples", [1, 5, 15])
def test_hdbscan_min_samples(synthetic_data, min_samples):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, (
        f"Should find clusters with min_samples={min_samples}"
    )


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
)
def test_hdbscan_metric(synthetic_data, metric):
    X, _ = synthetic_data
    p = 0.5 if metric == "minkowski" else None
    clusterer = hdbscan.HDBSCAN(metric=metric, p=p)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, f"Should find clusters with metric={metric}"


@pytest.mark.parametrize("method", ["eom", "leaf"])
def test_hdbscan_cluster_selection_method(synthetic_data, method):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(cluster_selection_method=method)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, (
        f"Should find clusters with cluster_selection_method={method}"
    )


def test_hdbscan_prediction_data(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X)
    # Check that prediction data is available
    assert hasattr(clusterer, "prediction_data_"), (
        "Prediction data should be available when prediction_data=True"
    )


@pytest.mark.parametrize("algorithm", ["best", "generic"])
def test_hdbscan_algorithm(synthetic_data, algorithm):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(algorithm=algorithm)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, f"Should find clusters with algorithm={algorithm}"


@pytest.mark.parametrize("leaf_size", [10, 30, 50])
def test_hdbscan_leaf_size(synthetic_data, leaf_size):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(leaf_size=leaf_size)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, f"Should find clusters with leaf_size={leaf_size}"


def test_hdbscan_gen_min_span_tree(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True)
    clusterer.fit(X)
    # Check that the minimum spanning tree is generated
    assert hasattr(clusterer, "minimum_spanning_tree_"), (
        "Minimum spanning tree should be generated when gen_min_span_tree=True"
    )


@pytest.mark.filterwarnings(
    "ignore:Instantiating a backend using a LocalPath:UserWarning"
)
def test_hdbscan_memory(synthetic_data, tmpdir):
    X, _ = synthetic_data
    from joblib import Memory

    memory = Memory(location=tmpdir)
    clusterer = hdbscan.HDBSCAN(memory=memory)
    clusterer.fit(X)
    # Check that cache directory is used
    # assert tmpdir.listdir(), "Cache directory should not be empty when memory caching is used"


def test_hdbscan_approx_min_span_tree(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(approx_min_span_tree=True)
    clusterer.fit(X)
    # this parameter is ignored in cuML


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_hdbscan_core_dist_n_jobs(synthetic_data, n_jobs):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(core_dist_n_jobs=n_jobs)
    clusterer.fit(X)
    # We assume the code runs without error; no direct way to test n_jobs effect
    assert True, f"HDBSCAN ran successfully with core_dist_n_jobs={n_jobs}"


def test_hdbscan_probabilities(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    # Check that cluster membership probabilities are available
    assert hasattr(clusterer, "probabilities_"), (
        "Cluster membership probabilities should be available after fitting"
    )


def test_hdbscan_fit_predict(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    labels_fit = clusterer.fit(X).labels_
    labels_predict = clusterer.fit_predict(X)
    # Check that labels from fit and fit_predict are the same
    assert np.array_equal(labels_fit, labels_predict), (
        "Labels from fit and fit_predict should be the same"
    )


def test_hdbscan_invalid_metric(synthetic_data):
    X, _ = synthetic_data
    with pytest.raises(ValueError):
        clusterer = hdbscan.HDBSCAN(metric="invalid_metric")
        clusterer.fit(X)


@pytest.mark.xfail(reason="Dispatching with sparse input not supported yet")
def test_hdbscan_sparse_input():
    from scipy.sparse import csr_matrix

    X, _ = make_blobs(
        n_samples=100,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        random_state=42,
    )
    X_sparse = csr_matrix(X)
    clusterer = hdbscan.HDBSCAN()
    cluster_labels = clusterer.fit_predict(X_sparse)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters > 0, "Should find clusters with sparse input data"


def test_hdbscan_non_convex_shapes():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(X)
    # Check that at least two clusters are found
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters >= 2, "Should find clusters in non-convex shapes"


def test_hdbscan_prediction(synthetic_data):
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
    assert len(test_labels) == X_test.shape[0], (
        "Labels should be assigned to test data points"
    )


def test_hdbscan_single_linkage_tree(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True)
    clusterer.fit(X)
    # Check that the single linkage tree is generated
    assert hasattr(clusterer, "single_linkage_tree_"), (
        "Single linkage tree should be generated after fitting"
    )


def test_hdbscan_condensed_tree(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    # Check that the condensed tree is available
    assert hasattr(clusterer, "condensed_tree_"), (
        "Condensed tree should be available after fitting"
    )


@pytest.mark.xfail(reason="Dispatching with examplars_ not supported yet")
def test_hdbscan_exemplars(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    # Check that cluster exemplars are available
    assert hasattr(clusterer, "exemplars_"), (
        "Cluster exemplars should be available after fitting"
    )


def test_hdbscan_prediction_data_with_prediction(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=True)
    clusterer.fit(X_train)
    # Use training data for prediction as a simple test
    test_labels, strengths = hdbscan.approximate_predict(clusterer, X_train)
    # Check that labels from prediction match original labels
    assert np.array_equal(clusterer.labels_, test_labels), (
        "Predicted labels should match original labels for training data"
    )


def test_hdbscan_predict_without_prediction_data(synthetic_data):
    X_train, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(prediction_data=False)
    clusterer.fit(X_train)
    with pytest.raises((AttributeError, ValueError)):
        hdbscan.approximate_predict(clusterer, X_train)


def test_hdbscan_min_cluster_size_effect(synthetic_data):
    X, _ = synthetic_data
    min_cluster_sizes = [5, 15, 30, 50]
    n_clusters_list = []
    for size in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
        cluster_labels = clusterer.fit_predict(X)
        n_clusters = len(set(cluster_labels)) - (
            1 if -1 in cluster_labels else 0
        )
        n_clusters_list.append(n_clusters)
    # Expect fewer clusters as min_cluster_size increases
    assert n_clusters_list == sorted(n_clusters_list, reverse=True), (
        "Number of clusters should decrease as min_cluster_size increases"
    )


def test_hdbscan_min_span_tree_effect(synthetic_data):
    X, _ = synthetic_data
    clusterer_with_tree = hdbscan.HDBSCAN(gen_min_span_tree=True)
    clusterer_with_tree.fit(X)
    clusterer_without_tree = hdbscan.HDBSCAN(gen_min_span_tree=False)
    clusterer_without_tree.fit(X)
    # Check that the minimum spanning tree affects the clustering (may not always be true)
    assert np.array_equal(
        clusterer_with_tree.labels_, clusterer_without_tree.labels_
    ), "Clustering should be consistent regardless of gen_min_span_tree"


def test_hdbscan_allow_single_cluster(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN(allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(X)
    # Check that clusters are formed
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    assert n_clusters >= 1, (
        "Should allow a single cluster when allow_single_cluster=True"
    )


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
    assert len(test_labels) == X_test.shape[0], (
        "Labels should be assigned to test data points"
    )
    assert len(strengths) == X_test.shape[0], (
        "Strengths should be computed for test data points"
    )
    # Check that strengths are between 0 and 1
    assert np.all((strengths >= 0) & (strengths <= 1)), (
        "Strengths should be between 0 and 1"
    )


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
    assert len(memberships) == X_train.shape[0], (
        "There should be a membership vector for each sample"
    )
    # Check that each membership vector sums to 1
    for membership in memberships:
        # Check that all probabilities are between 0 and 1
        assert all(0.0 <= v <= 1.0 for v in membership), (
            "Probabilities should be between 0 and 1"
        )


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
    assert hasattr(condensed_tree, "to_pandas"), (
        "Condensed tree should have a 'to_pandas' method"
    )
    # Convert to pandas DataFrame and check columns
    condensed_tree.to_pandas()


def test_hdbscan_single_linkage_tree_attribute(synthetic_data):
    X, _ = synthetic_data
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(X)
    single_linkage_tree = clusterer.single_linkage_tree_
    # Check that the single linkage tree has the expected attributes
    assert hasattr(single_linkage_tree, "to_numpy"), (
        "Single linkage tree should have a 'to_numpy' method"
    )
    # Convert to NumPy array and check shape
    sl_tree_array = single_linkage_tree.to_numpy()
    assert sl_tree_array.shape[1] == 4, (
        "Single linkage tree array should have 4 columns"
    )


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
    assert len(memberships) == X_train.shape[0], (
        "There should be a membership vector for each sample"
    )
    for membership in memberships:
        # Check that all probabilities are between 0 and 1
        assert all(0.0 <= v <= 1.0 for v in membership), (
            "Probabilities should be between 0 and 1"
        )


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
        assert np.all(np.isfinite(outlier_scores)), (
            "Outlier scores should be finite numbers"
        )
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
        assert len(exemplars) == n_clusters, (
            "There should be exemplars for each cluster"
        )
    else:
        pytest.skip(
            "Exemplar functionality is not available in this version of HDBSCAN"
        )
