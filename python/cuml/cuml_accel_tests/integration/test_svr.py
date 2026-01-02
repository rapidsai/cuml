# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@pytest.fixture(scope="module")
def linear_X_y():
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.fixture(scope="module")
def sinusoid_X_y():
    rng = np.random.default_rng(42)
    N = 200
    X = np.sort(5 * rng.random((N, 1)), axis=0)
    y = np.sin(X).ravel()

    # add noise to targets
    y[::5] += 3 * (0.5 - rng.random(N // 5))
    return X, y


def test_svr_linear(linear_X_y):
    X, y = linear_X_y
    svr = SVR(kernel="linear").fit(X, y)
    assert svr.score(X, y) > 0.5


@pytest.mark.parametrize("kernel", ["poly", "rbf"])
def test_svr_sinusoid(sinusoid_X_y, kernel):
    X, y = sinusoid_X_y
    svr = SVR(kernel=kernel).fit(X, y)
    assert svr.score(X, y) > 0.5


def test_svr_precomputed(linear_X_y):
    """Test SVR with precomputed kernel matrix."""
    X, y = linear_X_y
    # Compute linear kernel matrix
    K = X @ X.T
    svr = SVR(kernel="precomputed").fit(K, y)
    assert svr.score(K, y) > 0.5


def test_svr_precomputed_train_test():
    """Test SVR precomputed kernel with separate train/test sets."""
    X, y = make_regression(
        n_samples=150,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )
    X = StandardScaler().fit_transform(X)
    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]

    # Compute kernel matrices
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T

    svr = SVR(kernel="precomputed").fit(K_train, y_train)
    # Just check it runs and produces reasonable output
    predictions = svr.predict(K_test)
    assert predictions.shape == y_test.shape
