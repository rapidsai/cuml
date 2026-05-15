#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import types

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy as sp
import sklearn
from packaging.version import Version
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0.dev0")

requires_sklearn_18 = pytest.mark.skipif(
    not SKLEARN_18,
    reason="scikit-learn >= 1.8 required for StandardScaler acceleration",
)


class MockMethod:
    """A simple mock for method types, properly handling method binding"""

    def __init__(self, method):
        self._method = method

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return types.MethodType(self, obj)

    def __call__(self, *args, **kwargs):
        self.args = args[1:]  # drop self
        self.kwargs = kwargs
        return self._method(*args, **kwargs)


@pytest.fixture
def patch_methods(monkeypatch):
    """A fixture for patching one or more methods on a class"""

    def patch(cls, *methods):
        for method in methods:
            monkeypatch.setattr(cls, method, MockMethod(getattr(cls, method)))

    return patch


class HostTransformer(BaseEstimator):
    """A no-op host-only transformer"""

    def fit(self, X, y=None):
        assert isinstance(X, np.ndarray)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        return X

    def inverse_transform(self, X):
        assert isinstance(X, np.ndarray)
        return X


@pytest.fixture
def classification_data():
    # Create a synthetic dataset for binary classification
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    # Create a synthetic dataset for regression
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


classification_estimators = [
    LogisticRegression(),
    KNeighborsClassifier(),
]

regression_estimators = [
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    KNeighborsRegressor(),
]


@pytest.mark.parametrize(
    "transformer",
    [
        PCA(n_components=5),
        TruncatedSVD(n_components=5),
        KMeans(n_clusters=5, random_state=42),
    ],
)
@pytest.mark.parametrize("estimator", classification_estimators)
def test_classification_transformers(
    transformer, estimator, classification_data
):
    X_train, X_test, y_train, y_test = classification_data
    # Create pipeline with the transformer and estimator
    pipeline = Pipeline(
        [("transformer", transformer), ("classifier", estimator)]
    )
    # Fit and predict
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)
    # Ensure that the result is binary or multiclass classification


@pytest.mark.parametrize(
    "transformer",
    [
        PCA(n_components=5),
        TruncatedSVD(n_components=5),
        KMeans(n_clusters=5, random_state=42),
    ],
)
@pytest.mark.parametrize("estimator", regression_estimators)
def test_regression_transformers(transformer, estimator, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    # Create pipeline with the transformer and estimator
    pipeline = Pipeline(
        [("transformer", transformer), ("regressor", estimator)]
    )
    # Fit and predict
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)


@pytest.mark.parametrize(
    "transformer",
    [
        PCA(n_components=5),
        TruncatedSVD(n_components=5),
        KMeans(n_clusters=5, random_state=42),
    ],
)
@pytest.mark.parametrize("estimator", [NearestNeighbors(), DBSCAN()])
def test_unsupervised_neighbors(transformer, estimator, classification_data):
    X_train, X_test, _, _ = classification_data
    # Create pipeline with the transformer and unsupervised model
    pipeline = Pipeline(
        [("transformer", transformer), ("unsupervised", estimator)]
    )
    # Fit the model (no predict needed for unsupervised learning)
    pipeline.fit(X_train)


def test_umap_with_logistic_regression(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    # Create pipeline with UMAP for dimensionality reduction and logistic regression
    pipeline = Pipeline(
        [
            ("umap", UMAP(n_components=5, random_state=42)),
            ("classifier", LogisticRegression()),
        ]
    )
    # Fit and predict
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)


def test_automatic_step_naming():
    # The automatically generated names of estimators should be the
    # same with and without accelerator.
    pipeline = make_pipeline(PCA(), LogisticRegression())

    assert "pca" in pipeline.named_steps
    assert "logisticregression" in pipeline.named_steps


def test_pipeline_adding_none_value_as_labels(classification_data):
    X_train, _, _, _ = classification_data
    X_train = sp.sparse.csr_matrix(X_train)

    # Since cuML's TruncatedSVD does not handle sparse data,
    # the task will be automatically dispatched to Scikit-Learn.
    # If no labels are provided, Scikit-Learn's pipeline adds
    # y=None as the default labels.

    pipeline = make_pipeline(TruncatedSVD(n_components=20))
    pipeline.fit_transform(X_train)


@pytest.mark.parametrize(
    "order, enabled",
    [
        ("host", False),
        ("host-host", False),
        ("device-host", False),
        ("device", True),
        ("device-device", True),
        ("host-device", True),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
@requires_sklearn_18
def test_pipeline_data_transfer(
    order, enabled, nested, regression_data, patch_methods
):
    patch_methods(Ridge, "fit", "predict")
    X_train, X_test, y_train, y_test = regression_data
    xp = cp if enabled else np

    steps = [
        StandardScaler() if step == "device" else HostTransformer()
        for step in order.split("-")
    ]
    if nested:
        pipeline = make_pipeline(make_pipeline(*steps), Ridge())
    else:
        pipeline = make_pipeline(*steps, Ridge())

    pipeline.fit(X_train, y_train)
    assert isinstance(Ridge.fit.args[0], xp.ndarray)
    out = pipeline.predict(X_test)
    assert isinstance(Ridge.predict.args[0], xp.ndarray)
    # User-facing output is always numpy
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize(
    "pipeline, scaler, pca",
    [
        (
            make_pipeline(StandardScaler(), PCA()),
            (False, True),
            (True, False),
        ),
        (
            make_pipeline(HostTransformer(), StandardScaler(), PCA()),
            (False, False),
            (True, False),
        ),
        (
            make_pipeline(StandardScaler(), HostTransformer(), PCA()),
            (False, False),
            (False, False),
        ),
        (
            make_pipeline(StandardScaler(), PCA(), HostTransformer()),
            (False, True),
            (False, False),
        ),
        (
            make_pipeline(
                make_pipeline(HostTransformer(), StandardScaler()), PCA()
            ),
            (False, False),
            (True, False),
        ),
        (
            make_pipeline(
                make_pipeline(StandardScaler(), PCA()), HostTransformer()
            ),
            (False, True),
            (True, False),
        ),
    ],
)
@requires_sklearn_18
def test_pipeline_transform_data_transfer(
    pipeline, scaler, pca, regression_data, patch_methods
):
    patch_methods(StandardScaler, "transform", "inverse_transform")
    patch_methods(PCA, "transform", "inverse_transform")
    X = regression_data[0]

    pipeline.fit(X)

    def on_device(method, enabled):
        xp = cp if enabled else np
        return isinstance(method.args[0], xp.ndarray)

    out = pipeline.transform(X)
    assert isinstance(out, np.ndarray)
    assert on_device(PCA.transform, pca[0])
    assert on_device(StandardScaler.transform, scaler[0])

    out = pipeline.inverse_transform(X)
    assert isinstance(out, np.ndarray)
    assert on_device(PCA.inverse_transform, pca[1])
    assert on_device(StandardScaler.inverse_transform, scaler[1])


@requires_sklearn_18
def test_pipeline_data_transfer_with_host_fallback(
    regression_data, patch_methods
):
    """Intermediates passed on device, but step falls back to CPU for other reasons.

    Smoketests that the proxy converts device->host before fallback is called."""
    patch_methods(Ridge, "fit", "predict")
    X_train, X_test, y_train, y_test = regression_data

    pipeline = make_pipeline(StandardScaler(), Ridge(positive=True))
    pipeline.fit(X_train, y_train)
    assert isinstance(Ridge.fit.args[0], cp.ndarray)
    out = pipeline.predict(X_test)
    assert isinstance(Ridge.predict.args[0], cp.ndarray)
    # User-facing output is always numpy
    assert isinstance(out, np.ndarray)


@requires_sklearn_18
def test_pipeline_set_output():
    X, _ = make_regression(random_state=42)
    X2 = make_pipeline(
        StandardScaler().set_output(transform="pandas")
    ).fit_transform(X)
    assert isinstance(X2, pd.DataFrame)


@requires_sklearn_18
def test_pipeline_classifier_predict_non_numeric_labels(patch_methods):
    X, y = make_classification(random_state=42, n_classes=2)
    y = np.array(["a", "b"]).take(y)

    patch_methods(LogisticRegression, "fit", "predict")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline.fit(X, y)
    assert isinstance(LogisticRegression.fit.args[0], cp.ndarray)
    out = pipeline.predict(X)
    assert isinstance(LogisticRegression.predict.args[0], cp.ndarray)
    # User-facing output is always numpy
    assert isinstance(out, np.ndarray)
