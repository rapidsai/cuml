#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import types

import cupy as cp
import numpy as np
import pytest
import sklearn
from packaging.version import Version
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cuml.accel.estimator_proxy import is_proxy

AT_LEAST_SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0")


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


@pytest.mark.skipif(
    not AT_LEAST_SKLEARN_18,
    reason="GridSearchCV array API optimization requires sklearn >= 1.8",
)
def test_grid_search_data_on_device(regression_data, patch_methods):
    """Verify the patch sends cupy arrays to the inner estimator's fit."""
    patch_methods(Ridge, "fit")
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3)
    gs.fit(X, y)

    assert isinstance(Ridge.fit.args[0], cp.ndarray)


def _assert_no_cupy(gs):
    """Assert all fitted attributes on a GridSearchCV are host arrays."""
    for attr in ("best_score_", "best_index_"):
        val = getattr(gs, attr, None)
        if val is not None:
            assert not isinstance(val, cp.ndarray), (
                f"{attr} is cupy ({type(val)})"
            )

    for key, val in gs.cv_results_.items():
        assert not isinstance(val, cp.ndarray), f"cv_results_[{key!r}] is cupy"


def test_grid_search_output_types(regression_data):
    """All user-facing outputs are numpy, not cupy."""
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3, scoring="r2")
    gs.fit(X, y)

    _assert_no_cupy(gs)

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
    _assert_no_cupy(gs)


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
    _assert_no_cupy(gs)


def test_grid_search_refit_callable(regression_data):
    """refit=callable receives numpy cv_results_ and works correctly."""

    def my_refit(cv_results):
        # A typical user callable that uses numpy operations
        scores = cv_results["mean_test_score"]
        assert isinstance(scores, np.ndarray), (
            f"Expected numpy in refit callable, got {type(scores)}"
        )
        return np.argmax(scores)

    X, y = regression_data
    gs = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="r2",
        refit=my_refit,
    )
    gs.fit(X, y)

    assert gs.best_index_ is not None
    assert is_proxy(gs.best_estimator_)
    _assert_no_cupy(gs)


def test_grid_search_custom_scorer(regression_data):
    """Custom scorers receive numpy arrays, not cupy."""
    scorer_arg_types = []

    def my_metric(y_true, y_pred):
        scorer_arg_types.append((type(y_true).__name__, type(y_pred).__name__))
        return -np.mean((y_true - y_pred) ** 2)

    X, y = regression_data
    gs = GridSearchCV(
        Ridge(),
        {"alpha": [0.1, 1.0]},
        cv=3,
        scoring=make_scorer(my_metric),
    )
    gs.fit(X, y)

    assert all(
        yt == "ndarray" and yp == "ndarray" for yt, yp in scorer_arg_types
    ), f"Expected all numpy in scorer, got {scorer_arg_types}"
    assert not np.isnan(gs.best_score_)


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


def test_grid_search_n_jobs_skips_optimization(regression_data, patch_methods):
    """n_jobs != 1 skips GPU optimization (thread-local state doesn't propagate)."""
    patch_methods(Ridge, "fit")

    X, y = regression_data
    gs = GridSearchCV(
        Ridge(), {"alpha": [0.1, 1.0]}, cv=3, scoring="r2", n_jobs=2
    )
    gs.fit(X, y)

    assert isinstance(Ridge.fit.args[0], np.ndarray), (
        "Expected numpy (optimization should be skipped for n_jobs>1)"
    )
    assert not np.isnan(gs.best_score_)


def test_grid_search_all_params_unsupported(regression_data):
    """When no param combo supports GPU, falls back to numpy path."""
    X, y = regression_data

    gs = GridSearchCV(Ridge(), {"positive": [True]}, cv=3, scoring="r2")
    gs.fit(X, y)

    assert not np.isnan(gs.best_score_)


def test_grid_search_string_labels(patch_methods):
    """String y labels bail out to unoptimized path (cupy can't hold strings)."""
    patch_methods(LogisticRegression, "fit")

    X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
    y = np.array(["cat", "dog"] * 50)

    gs = GridSearchCV(
        LogisticRegression(max_iter=200),
        {"C": [0.1, 1.0]},
        cv=3,
        scoring="accuracy",
    )
    gs.fit(X, y)

    assert isinstance(LogisticRegression.fit.args[0], np.ndarray), (
        "Expected numpy (optimization should be skipped for string labels)"
    )
    assert gs.best_score_ > 0


def test_grid_search_pipeline_non_proxy_tail(regression_data):
    """Pipeline with a non-proxy tail step skips GPU optimization.

    When the Pipeline's tail step is not a proxy, predictions will be
    numpy while array-API CV splitting keeps y_test as cupy, causing a
    device mismatch in scoring. Need to fall back to the unoptimized path.
    """
    fit_X_types = []

    class SimpleRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            fit_X_types.append(type(X))
            self.mean_ = np.mean(y)
            return self

        def predict(self, X):
            return np.full(X.shape[0], self.mean_)

    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", SimpleRegressor())])

    gs = GridSearchCV(
        pipe, {"scaler__with_mean": [True, False]}, cv=3, scoring="r2"
    )
    gs.fit(X, y)

    assert all(t is np.ndarray for t in fit_X_types), (
        f"Expected all numpy fits (optimization skipped), got {fit_X_types}"
    )
    assert not np.isnan(gs.best_score_)


@pytest.mark.skipif(
    not AT_LEAST_SKLEARN_18,
    reason="GridSearchCV array API optimization requires sklearn >= 1.8",
)
def test_grid_search_pipeline_all_proxy(regression_data, patch_methods):
    """Pipeline with all proxy steps uses the cupy optimization path."""
    patch_methods(Ridge, "fit")

    X, y = regression_data
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])

    gs = GridSearchCV(
        pipe, {"ridge__alpha": [0.1, 1.0, 10.0]}, cv=3, scoring="r2"
    )
    gs.fit(X, y)

    assert isinstance(Ridge.fit.args[0], cp.ndarray), (
        "Expected cupy (optimization should be active)"
    )
    _assert_no_cupy(gs)
