# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Pytest tests for cuML's IsolationForest implementation.

These tests are designed to be:
1. Complete - covering all API methods and parameters
2. Fast-running - using small datasets for unit tests
3. Based on sklearn IsolationForest test patterns
4. Validating sklearn compatibility
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest as skIsolationForest

from cuml import IsolationForest as cuIsolationForest
from cuml.testing.utils import stress_param, unit_param


# =============================================================================
# Fixtures for test data
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_data_small():
    """Small dataset for fast unit tests."""
    rng = np.random.RandomState(42)
    # Normal data centered at origin
    X_normal = rng.randn(100, 4).astype(np.float32)
    # Outliers far from origin
    X_outliers = rng.uniform(low=-6, high=6, size=(10, 4)).astype(np.float32)
    X = np.vstack([X_normal, X_outliers])
    # Labels: 1 for normal, -1 for outliers (sklearn convention)
    y_true = np.array([1] * 100 + [-1] * 10)
    return X, y_true


@pytest.fixture(scope="module")
def blobs_data():
    """Blob dataset for clustering-like anomaly detection."""
    X, _ = make_blobs(
        n_samples=200,
        n_features=4,
        centers=2,
        cluster_std=1.0,
        random_state=42,
    )
    return X.astype(np.float32)


# =============================================================================
# Basic functionality tests
# =============================================================================


def test_fit_returns_self(synthetic_data_small):
    """fit() should return self for method chaining."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    result = clf.fit(X)
    assert result is clf


def test_fit_predict(synthetic_data_small):
    """fit_predict() should work correctly."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    predictions = clf.fit_predict(X)
    assert predictions.shape == (X.shape[0],)
    assert set(np.unique(predictions)).issubset({-1, 1})


def test_predict_returns_correct_shape(synthetic_data_small):
    """predict() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    predictions = clf.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_predict_returns_valid_labels(synthetic_data_small):
    """predict() should return only -1 (anomaly) or 1 (normal)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    predictions = clf.predict(X)
    unique_labels = set(np.unique(predictions))
    assert unique_labels.issubset({-1, 1})


def test_score_samples_returns_correct_shape(synthetic_data_small):
    """score_samples() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_decision_function_returns_correct_shape(synthetic_data_small):
    """decision_function() should return array of shape (n_samples,)."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    df = clf.decision_function(X)
    assert df.shape == (X.shape[0],)


def test_n_features_in_set_after_fit(synthetic_data_small):
    """n_features_in_ should be set after fit."""
    X, _ = synthetic_data_small
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    assert clf.n_features_in_ == X.shape[1]


# =============================================================================
# Parameter handling tests
# =============================================================================


def test_default_parameters():
    """Default parameters should match expected values."""
    clf = cuIsolationForest()
    assert clf.n_estimators == 100
    assert clf.max_samples == 256
    assert clf.max_depth is None
    assert clf.max_features == 1.0
    assert clf.bootstrap is False
    assert clf.contamination == "auto"


@pytest.mark.parametrize("n_estimators", [10, 50, 100])
def test_n_estimators_parameter(blobs_data, n_estimators):
    """n_estimators parameter should be respected."""
    clf = cuIsolationForest(n_estimators=n_estimators, random_state=42)
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_samples", [32, 64, 128, 256])
def test_max_samples_int(blobs_data, max_samples):
    """Integer max_samples should subsample correctly."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=max_samples, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_samples", [0.5, 0.8, 1.0])
def test_max_samples_float(blobs_data, max_samples):
    """Float max_samples should work as fraction."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples=max_samples, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


def test_max_samples_auto(blobs_data):
    """max_samples='auto' should use min(256, n_samples)."""
    clf = cuIsolationForest(
        n_estimators=10, max_samples="auto", random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_depth", [2, 4, 8, None])
def test_max_depth_parameter(blobs_data, max_depth):
    """max_depth parameter should be respected."""
    clf = cuIsolationForest(
        n_estimators=10, max_depth=max_depth, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("max_features", [0.5, 0.8, 1.0])
def test_max_features_parameter(blobs_data, max_features):
    """max_features parameter should be respected."""
    clf = cuIsolationForest(
        n_estimators=10, max_features=max_features, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


@pytest.mark.parametrize("bootstrap", [True, False])
def test_bootstrap_parameter(blobs_data, bootstrap):
    """bootstrap parameter should be respected."""
    clf = cuIsolationForest(
        n_estimators=10, bootstrap=bootstrap, random_state=42
    )
    clf.fit(blobs_data)
    predictions = clf.predict(blobs_data)
    assert predictions.shape[0] == blobs_data.shape[0]


# =============================================================================
# Data type tests
# =============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_dtypes(dtype):
    """Should handle float32 and float64."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4).astype(dtype)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)

    predictions = clf.predict(X)
    assert predictions.shape == (X.shape[0],)


# =============================================================================
# Anomaly detection quality tests
# =============================================================================


def test_outliers_score_lower_sklearn_convention(synthetic_data_small):
    """
    Outliers should have lower (more negative) scores than inliers.
    In sklearn convention: more negative = more anomalous.
    """
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    scores = np.asarray(scores)

    inlier_mask = y_true == 1
    outlier_mask = y_true == -1
    mean_inlier_score = np.mean(scores[inlier_mask])
    mean_outlier_score = np.mean(scores[outlier_mask])

    assert mean_outlier_score < mean_inlier_score, (
        f"Outliers should have lower scores than inliers. "
        f"Got inlier={mean_inlier_score:.4f}, outlier={mean_outlier_score:.4f}"
    )


def test_outliers_detected_by_predict(synthetic_data_small):
    """predict() should detect at least some outliers."""
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    predictions = clf.predict(X)
    predictions = np.asarray(predictions)

    n_detected = np.sum(predictions == -1)
    assert n_detected > 0, "Should detect at least some anomalies"


def test_extreme_outliers_detected():
    """Extreme outliers should be reliably detected."""
    rng = np.random.RandomState(42)
    X_normal = rng.randn(200, 4).astype(np.float32)
    X_outlier = np.array([[100, 100, 100, 100]], dtype=np.float32)
    X = np.vstack([X_normal, X_outlier])

    clf = cuIsolationForest(n_estimators=100, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    scores = np.asarray(scores)

    outlier_score = scores[-1]
    normal_scores = scores[:-1]
    assert outlier_score < np.percentile(
        normal_scores, 5
    ), "Extreme outlier should have very low score"


# =============================================================================
# sklearn compatibility tests
# =============================================================================


def test_api_compatibility(blobs_data):
    """cuML IF should have the same API as sklearn IF."""
    cu_clf = cuIsolationForest(n_estimators=10, random_state=42)
    sk_clf = skIsolationForest(n_estimators=10, random_state=42)

    cu_clf.fit(blobs_data)
    sk_clf.fit(blobs_data)

    cu_preds = cu_clf.predict(blobs_data)
    sk_preds = sk_clf.predict(blobs_data)
    assert cu_preds.shape == sk_preds.shape

    cu_scores = cu_clf.score_samples(blobs_data)
    sk_scores = sk_clf.score_samples(blobs_data)
    assert cu_scores.shape == sk_scores.shape

    cu_df = cu_clf.decision_function(blobs_data)
    sk_df = sk_clf.decision_function(blobs_data)
    assert cu_df.shape == sk_df.shape


def test_score_convention_matches_sklearn(synthetic_data_small):
    """
    Score convention should match sklearn:
    - More negative scores = more anomalous
    - Outliers should have negative decision_function values
    """
    X, y_true = synthetic_data_small
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X)

    df = clf.decision_function(X)
    df = np.asarray(df)

    outlier_mask = y_true == -1
    inlier_mask = y_true == 1
    mean_outlier_df = np.mean(df[outlier_mask])
    mean_inlier_df = np.mean(df[inlier_mask])
    assert mean_outlier_df < mean_inlier_df


def test_similar_results_to_sklearn(blobs_data):
    """
    cuML IF should produce similar (not identical) results to sklearn.

    Note: Results won't be identical due to different RNG implementations,
    tree building algorithms, and GPU vs CPU computation.
    But the relative ordering should be similar.
    """
    X = blobs_data

    cu_clf = cuIsolationForest(n_estimators=100, random_state=42)
    sk_clf = skIsolationForest(n_estimators=100, random_state=42)

    cu_clf.fit(X)
    sk_clf.fit(X)

    cu_scores = np.asarray(cu_clf.score_samples(X))
    sk_scores = sk_clf.score_samples(X)

    correlation = np.corrcoef(cu_scores, sk_scores)[0, 1]
    assert correlation > 0.5, (
        f"Score correlation with sklearn should be > 0.5, got {correlation:.3f}"
    )


# =============================================================================
# Determinism tests
# =============================================================================


def test_same_random_state_same_results(blobs_data):
    """Same random_state should produce identical results."""
    clf1 = cuIsolationForest(n_estimators=20, random_state=42)
    clf2 = cuIsolationForest(n_estimators=20, random_state=42)

    clf1.fit(blobs_data)
    clf2.fit(blobs_data)

    scores1 = np.asarray(clf1.score_samples(blobs_data))
    scores2 = np.asarray(clf2.score_samples(blobs_data))

    np.testing.assert_array_almost_equal(
        scores1, scores2, decimal=5,
        err_msg="Same random_state should produce identical scores"
    )


def test_different_random_state_different_results(blobs_data):
    """Different random_state should produce different results."""
    clf1 = cuIsolationForest(n_estimators=20, random_state=42)
    clf2 = cuIsolationForest(n_estimators=20, random_state=123)

    clf1.fit(blobs_data)
    clf2.fit(blobs_data)

    scores1 = np.asarray(clf1.score_samples(blobs_data))
    scores2 = np.asarray(clf2.score_samples(blobs_data))

    assert not np.allclose(scores1, scores2), (
        "Different random_state should produce different scores"
    )


# =============================================================================
# Edge case tests
# =============================================================================


def test_small_dataset():
    """Should handle small datasets."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3).astype(np.float32)

    clf = cuIsolationForest(n_estimators=5, max_samples=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_single_feature():
    """Should handle single-feature data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1).astype(np.float32)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_many_features():
    """Should handle high-dimensional data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 50).astype(np.float32)

    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(X)
    scores = clf.score_samples(X)
    assert scores.shape == (X.shape[0],)


def test_predict_before_fit_raises():
    """predict() before fit() should raise an error."""
    clf = cuIsolationForest()
    X = np.random.randn(10, 3).astype(np.float32)

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.predict(X)


def test_score_samples_before_fit_raises():
    """score_samples() before fit() should raise an error."""
    clf = cuIsolationForest()
    X = np.random.randn(10, 3).astype(np.float32)

    with pytest.raises(RuntimeError, match="not been fitted"):
        clf.score_samples(X)


def test_feature_mismatch_raises(blobs_data):
    """Predicting with wrong number of features should raise."""
    clf = cuIsolationForest(n_estimators=10, random_state=42)
    clf.fit(blobs_data)

    X_wrong = np.random.randn(10, blobs_data.shape[1] + 1).astype(np.float32)

    with pytest.raises(ValueError, match="features"):
        clf.predict(X_wrong)


# =============================================================================
# Performance-oriented tests (parameterized by test level)
# =============================================================================


@pytest.mark.parametrize(
    "nrows", [unit_param(200), stress_param(50000)]
)
@pytest.mark.parametrize(
    "ncols", [unit_param(10), stress_param(200)]
)
@pytest.mark.parametrize("dtype", [np.float32])
def test_isolation_forest_scaling(nrows, ncols, dtype):
    """Test IF scales with data size."""
    rng = np.random.RandomState(42)
    X = rng.randn(nrows, ncols).astype(dtype)

    clf = cuIsolationForest(n_estimators=20, random_state=42)
    clf.fit(X)

    scores = clf.score_samples(X)
    assert scores.shape == (nrows,)

    predictions = clf.predict(X)
    assert predictions.shape == (nrows,)
    assert set(np.unique(predictions)).issubset({-1, 1})


@pytest.mark.parametrize("n_estimators", [unit_param(10), stress_param(50)])
def test_n_estimators_scaling(blobs_data, n_estimators):
    """Test IF with different numbers of estimators."""
    clf = cuIsolationForest(n_estimators=n_estimators, random_state=42)
    clf.fit(blobs_data)

    scores = clf.score_samples(blobs_data)
    predictions = clf.predict(blobs_data)

    assert scores.shape == (blobs_data.shape[0],)
    assert predictions.shape == (blobs_data.shape[0],)


# =============================================================================
# Integration tests
# =============================================================================


def test_full_workflow(synthetic_data_small):
    """Test complete anomaly detection workflow."""
    X, y_true = synthetic_data_small

    # Split data (don't use outliers in training for realistic scenario)
    X_train = X[y_true == 1]  # Only normal points for training
    X_test = X

    # Train model
    clf = cuIsolationForest(n_estimators=50, random_state=42)
    clf.fit(X_train)

    # Get predictions and scores
    predictions = clf.predict(X_test)
    scores = clf.score_samples(X_test)
    df = clf.decision_function(X_test)

    # Basic sanity checks
    assert predictions.shape == X_test.shape[:1]
    assert scores.shape == X_test.shape[:1]
    assert df.shape == X_test.shape[:1]

    # Check that model detected something
    predictions = np.asarray(predictions)
    n_anomalies = np.sum(predictions == -1)
    assert n_anomalies > 0, "Should detect some anomalies"


def test_train_on_pure_data_detect_outliers():
    """Train on pure data, then detect injected outliers."""
    rng = np.random.RandomState(42)

    # Pure training data (no outliers)
    X_train = rng.randn(500, 4).astype(np.float32)

    # Test data with injected outliers
    X_test_normal = rng.randn(100, 4).astype(np.float32)
    X_test_outliers = np.array([
        [10, 10, 10, 10],
        [-10, -10, -10, -10],
        [5, -5, 5, -5],
    ], dtype=np.float32)
    X_test = np.vstack([X_test_normal, X_test_outliers])

    # Train and predict
    clf = cuIsolationForest(n_estimators=100, random_state=42)
    clf.fit(X_train)
    predictions = clf.predict(X_test)
    predictions = np.asarray(predictions)

    # The injected outliers should be detected
    outlier_predictions = predictions[-3:]
    detected_outliers = np.sum(outlier_predictions == -1)
    assert detected_outliers >= 2, (
        f"Should detect at least 2 of 3 injected outliers, got {detected_outliers}"
    )
