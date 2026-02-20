#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import pytest
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
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


class _PassthroughTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer; used with _spy_on_transform to inspect pipeline data flow."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def _spy_on_transform(monkeypatch, estimator, received=None):
    """Patch the estimator's class to record the type of X passed to transform.

    Patches the class so that when the pipeline clones the step, the clone still
    uses the spy and receives the correct `self`.
    """
    if received is None:
        received = {}
    cls = type(estimator)
    orig_transform = cls.transform
    orig_fit_transform = getattr(cls, "fit_transform", None)

    def spied_transform(self, X, *args, **kwargs):
        received["X_type"] = type(X)
        return orig_transform(self, X, *args, **kwargs)

    def spied_fit_transform(self, X, y=None, *args, **kwargs):
        received["X_type"] = type(X)
        return orig_fit_transform(self, X, y, *args, **kwargs)

    monkeypatch.setattr(cls, "transform", spied_transform)
    if orig_fit_transform is not None:
        monkeypatch.setattr(cls, "fit_transform", spied_fit_transform)
    return estimator


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


def test_pipeline_intermediate_data_stays_on_device(monkeypatch):
    """Between consecutive accelerated steps, data must stay on GPU (cupy)."""
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    received = {}
    pca = _spy_on_transform(monkeypatch, PCA(n_components=5), received)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", pca),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)
    assert received.get("X_type") is cp.ndarray


def test_pipeline_non_accelerated_step_receives_numpy(monkeypatch):
    """Non-accelerated step after an accelerated step should receive numpy."""
    X, _ = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    received = {}
    check = _spy_on_transform(monkeypatch, _PassthroughTransformer(), received)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("check", check),
            ("pca", PCA(n_components=5)),
        ]
    )
    pipeline.fit_transform(X)
    assert received.get("X_type") is not cp.ndarray


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
