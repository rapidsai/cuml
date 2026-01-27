# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def multiclass():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


def test_svc(binary):
    X, y = binary
    svc = SVC().fit(X, y)
    assert svc.score(X, y) > 0.5


def test_svc_probability(binary):
    X, y = binary
    svc = SVC(probability=True).fit(X, y)
    # Inference and score works
    assert svc.score(X, y) > 0.5

    # probA_ and probB_ exist and are the correct shape
    assert svc.probA_.shape == (1,)
    assert svc.probB_.shape == (1,)

    # predict_proba works
    y_pred = svc.predict_proba(X).argmax(axis=1)
    assert accuracy_score(y_pred, y) > 0.5


def test_svc_multiclass(multiclass):
    X, y = multiclass
    svr = SVC().fit(X, y)
    assert svr.score(X, y) > 0.5


def test_conditional_methods():
    svc = SVC()
    assert not hasattr(svc, "predict_proba")
    assert not hasattr(svc, "predict_log_proba")

    svc.probability = True
    assert hasattr(svc, "predict_proba")
    assert hasattr(svc, "predict_log_proba")
    # Ensure these methods aren't forwarded CPU attributes
    assert svc.predict_proba is not svc._cpu.predict_proba
    assert svc.predict_log_proba is not svc._cpu.predict_log_proba


def test_svc_precomputed(binary):
    """Test SVC with precomputed kernel matrix."""
    X, y = binary
    # Compute linear kernel matrix
    K = X @ X.T
    svc = SVC(kernel="precomputed").fit(K, y)
    assert svc.score(K, y) > 0.5


def test_svc_precomputed_train_test():
    """Test SVC precomputed kernel with separate train/test sets."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_classes=2,
        n_informative=5,
        random_state=42,
    )
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]

    # Compute kernel matrices
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T

    svc = SVC(kernel="precomputed").fit(K_train, y_train)
    accuracy = accuracy_score(y_test, svc.predict(K_test))
    assert accuracy > 0.5
