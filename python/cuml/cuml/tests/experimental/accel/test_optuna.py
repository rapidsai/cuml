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
import optuna
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
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
import umap
import hdbscan


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=100, n_features=10, noise=0.1, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial, estimator, X_train, y_train):
    params = {}
    if hasattr(estimator, "C"):
        params["C"] = trial.suggest_loguniform("C", 1e-3, 1e2)
    if hasattr(estimator, "alpha"):
        params["alpha"] = trial.suggest_loguniform("alpha", 1e-3, 1e2)
    if hasattr(estimator, "l1_ratio"):
        params["l1_ratio"] = trial.suggest_uniform("l1_ratio", 0.0, 1.0)
    if hasattr(estimator, "n_neighbors"):
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 15)
    model = estimator.set_params(**params)
    score = cross_val_score(model, X_train, y_train, cv=3).mean()
    return score


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        KNeighborsClassifier(),
    ],
)
def test_classification_models_optuna(estimator, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, estimator, X_train, y_train),
        n_trials=10,
    )

    assert study.best_value > 0.5, f"Failed to optimize {estimator}"


@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression(),
        Ridge(),
        Lasso(),
        ElasticNet(),
        KernelRidge(),
        KNeighborsRegressor(),
    ],
)
def test_regression_models_optuna(estimator, regression_data):
    X_train, X_test, y_train, y_test = regression_data
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, estimator, X_train, y_train),
        n_trials=10,
    )
    assert study.best_value < 1.0, f"Failed to optimize {estimator}"


@pytest.mark.parametrize(
    "clustering_method",
    [
        KMeans(n_clusters=3, random_state=42),
        DBSCAN(),
        hdbscan.HDBSCAN(min_cluster_size=5),
    ],
)
def test_clustering_models(clustering_method, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clustering_method.fit(X_train)
    assert True, f"{clustering_method} successfully ran"


@pytest.mark.parametrize(
    "dimensionality_reduction_method",
    [
        PCA(n_components=5),
        TruncatedSVD(n_components=5),
        umap.UMAP(n_components=5),
        TSNE(n_components=2),
    ],
)
def test_dimensionality_reduction(
    dimensionality_reduction_method, classification_data
):
    X_train, X_test, y_train, y_test = classification_data
    X_transformed = dimensionality_reduction_method.fit_transform(X_train)
    assert (
        X_transformed.shape[1] <= 5
    ), f"{dimensionality_reduction_method} successfully reduced dimensions"


def test_nearest_neighbors(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    nearest_neighbors = NearestNeighbors(n_neighbors=5)
    nearest_neighbors.fit(X_train)
    assert True, "NearestNeighbors successfully ran"
