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
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("n_neighbors", [1, 3, 5, 10])
def test_knn_regressor_n_neighbors(regression_data, n_neighbors):
    X, y = regression_data
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_score(y, y_pred)


@pytest.mark.parametrize("weights", ["uniform"])
def test_knn_regressor_weights(regression_data, weights):
    X, y = regression_data
    model = KNeighborsRegressor(weights=weights)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.7, f"R^2 score should be reasonable with weights={weights}"


@pytest.mark.parametrize(
    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
)
def test_knn_regressor_algorithm(regression_data, algorithm):
    X, y = regression_data
    model = KNeighborsRegressor(algorithm=algorithm)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.7
    ), f"R^2 score should be reasonable with algorithm={algorithm}"


@pytest.mark.parametrize("leaf_size", [10, 30, 50])
def test_knn_regressor_leaf_size(regression_data, leaf_size):
    X, y = regression_data
    model = KNeighborsRegressor(leaf_size=leaf_size)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.7
    ), f"R^2 score should be reasonable with leaf_size={leaf_size}"


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
)
def test_knn_regressor_metric(regression_data, metric):
    X, y = regression_data
    model = KNeighborsRegressor(metric=metric)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_score(y, y_pred)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_knn_regressor_p_parameter(regression_data, p):
    X, y = regression_data
    model = KNeighborsRegressor(metric="minkowski", p=p)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.7, f"R^2 score should be reasonable with p={p}"


@pytest.mark.xfail(reason="Dispatching with callable not supported yet")
def test_knn_regressor_weights_callable(regression_data):
    X, y = regression_data

    def custom_weights(distances):
        return np.ones_like(distances)

    model = KNeighborsRegressor(weights=custom_weights)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.7, "R^2 score should be reasonable with custom weights"


@pytest.mark.xfail(
    reason="cuML and sklearn don't have matching exceptions yet"
)
def test_knn_regressor_invalid_algorithm(regression_data):
    X, y = regression_data
    with pytest.raises((ValueError, KeyError)):
        model = KNeighborsRegressor(algorithm="invalid_algorithm")
        model.fit(X, y)


@pytest.mark.xfail(
    reason="cuML and sklearn don't have matching exceptions yet"
)
def test_knn_regressor_invalid_metric(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError):
        model = KNeighborsRegressor(metric="invalid_metric")
        model.fit(X, y)


def test_knn_regressor_invalid_weights(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError):
        model = KNeighborsRegressor(weights="invalid_weight")
        model.fit(X, y)


def test_knn_regressor_sparse_input():
    from scipy.sparse import csr_matrix

    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_sparse = csr_matrix(X)
    model = KNeighborsRegressor()
    model.fit(X_sparse, y)
    y_pred = model.predict(X_sparse)
    r2_score(y, y_pred)


def test_knn_regressor_multioutput():
    X, y = make_regression(
        n_samples=100, n_features=20, n_targets=3, random_state=42
    )
    model = KNeighborsRegressor()
    model.fit(X, y)
    y_pred = model.predict(X)
    # Check that the predicted shape matches the true targets
    assert (
        y_pred.shape == y.shape
    ), "Predicted outputs should have the same shape as true outputs"
    # Calculate R^2 score for multi-output regression
    r2_score(y, y_pred)
