#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import types

import cupy as cp
import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV

from cuml.accel.estimator_proxy import is_proxy


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    return X, y


# XXX Time for some kind of utils module to share this?
class MockMethod:
    def __init__(self, method):
        self._method = method

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return types.MethodType(self, obj)

    def __call__(self, *args, **kwargs):
        self.args = args[1:]
        self.kwargs = kwargs
        return self._method(*args, **kwargs)


@pytest.fixture
def patch_methods(monkeypatch):
    def patch(cls, *methods):
        for method in methods:
            monkeypatch.setattr(cls, method, MockMethod(getattr(cls, method)))

    return patch


def test_grid_search_basic(regression_data):
    """GridSearchCV with a proxy estimator produces correct results."""
    X, y = regression_data
    assert is_proxy(Ridge())

    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=3)
    gs.fit(X, y)

    assert gs.best_params_ is not None
    assert isinstance(gs.best_score_, (float, np.floating))
    assert not np.isnan(gs.best_score_)
    assert is_proxy(gs.best_estimator_)


def test_grid_search_data_on_device(regression_data, patch_methods):
    """Verify the patch sends cupy arrays to the inner estimator's fit."""
    patch_methods(Ridge, "fit")
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3)
    gs.fit(X, y)

    assert isinstance(Ridge.fit.args[0], cp.ndarray)


def test_grid_search_output_types(regression_data):
    """All user-facing outputs are numpy, not cupy."""
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3, scoring="r2")
    gs.fit(X, y)

    assert isinstance(gs.best_score_, (float, np.floating))
    assert isinstance(gs.cv_results_["mean_test_score"], np.ndarray)

    pred = gs.predict(X[:5])
    assert isinstance(pred, np.ndarray)

    score = gs.score(X, y)
    assert isinstance(score, (float, np.floating))


def test_grid_search_mixed_gpu_cpu_fallback(regression_data):
    """Param combos that fall back to CPU still produce valid scores."""
    X, y = regression_data

    gs = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0], "positive": [True, False]},
        cv=3,
        scoring="r2",
    )
    gs.fit(X, y)

    scores = gs.cv_results_["mean_test_score"]
    assert len(scores) == 4
    assert not any(np.isnan(scores)), f"NaN scores found: {scores}"


def test_grid_search_multimetric(regression_data):
    """Multimetric scoring works."""
    X, y = regression_data

    gs = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0]},
        cv=3,
        scoring=["r2", "neg_mean_squared_error"],
        refit="r2",
    )
    gs.fit(X, y)

    assert "mean_test_r2" in gs.cv_results_
    assert "mean_test_neg_mean_squared_error" in gs.cv_results_
    assert isinstance(gs.cv_results_["mean_test_r2"], np.ndarray)


def test_grid_search_classification(classification_data):
    """GridSearchCV works with classification estimators."""
    X, y = classification_data

    gs = GridSearchCV(
        LogisticRegression(max_iter=200),
        {"C": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="accuracy",
    )
    gs.fit(X, y)

    assert gs.best_score_ > 0
    assert is_proxy(gs.best_estimator_)


def test_grid_search_non_proxy_estimator(regression_data):
    """Non-proxy estimators skip the GPU optimization."""

    class SimpleRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            assert isinstance(X, np.ndarray), "Expected numpy, got cupy"
            self.mean_ = np.mean(y)
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.mean_)

    X, y = regression_data
    assert not is_proxy(SimpleRegressor())

    gs = GridSearchCV(SimpleRegressor(), {}, cv=3)
    gs.fit(X, y)
    assert isinstance(gs.best_score_, (float, np.floating))


def test_grid_search_all_params_unsupported(regression_data):
    """When no param combo supports GPU, falls back to numpy path."""
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"positive": [True]}, cv=3, scoring="r2")
    gs.fit(X, y)

    assert not np.isnan(gs.best_score_)
