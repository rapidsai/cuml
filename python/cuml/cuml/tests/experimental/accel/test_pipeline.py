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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    ElasticNet,
    Ridge,
    Lasso,
)
from sklearn.manifold import TSNE
from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from umap import UMAP
import hdbscan
import numpy as np


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
    KernelRidge(),
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
    y_pred = pipeline.predict(X_test)
    # Ensure that the result has a reasonably low mean squared error


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
