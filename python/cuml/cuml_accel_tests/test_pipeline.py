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

import pytest
import scipy as sp
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
from umap import UMAP


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
