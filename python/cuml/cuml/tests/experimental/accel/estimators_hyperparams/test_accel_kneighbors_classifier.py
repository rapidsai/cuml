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
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("n_neighbors", [1, 3, 5, 10])
def test_knn_classifier_n_neighbors(classification_data, n_neighbors):
    X, y = classification_data
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert (
        acc > 0.7
    ), f"Accuracy should be reasonable with n_neighbors={n_neighbors}"


@pytest.mark.parametrize("weights", ["uniform"])
def test_knn_classifier_weights(classification_data, weights):
    X, y = classification_data
    model = KNeighborsClassifier(weights=weights)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, f"Accuracy should be reasonable with weights={weights}"


@pytest.mark.parametrize(
    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
)
def test_knn_classifier_algorithm(classification_data, algorithm):
    X, y = classification_data
    model = KNeighborsClassifier(algorithm=algorithm)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert (
        acc > 0.7
    ), f"Accuracy should be reasonable with algorithm={algorithm}"


@pytest.mark.parametrize("leaf_size", [10, 30, 50])
def test_knn_classifier_leaf_size(classification_data, leaf_size):
    X, y = classification_data
    model = KNeighborsClassifier(leaf_size=leaf_size)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert (
        acc > 0.7
    ), f"Accuracy should be reasonable with leaf_size={leaf_size}"


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"]
)
def test_knn_classifier_metric(classification_data, metric):
    X, y = classification_data
    model = KNeighborsClassifier(metric=metric)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, f"Accuracy should be reasonable with metric={metric}"


@pytest.mark.parametrize("p", [1, 2, 3])
def test_knn_classifier_p_parameter(classification_data, p):
    X, y = classification_data
    model = KNeighborsClassifier(metric="minkowski", p=p)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, f"Accuracy should be reasonable with p={p}"


@pytest.mark.xfail(reason="Dispatching with callable not supported yet")
def test_knn_classifier_weights_callable(classification_data):
    X, y = classification_data

    def custom_weights(distances):
        return np.ones_like(distances)

    model = KNeighborsClassifier(weights=custom_weights)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, "Accuracy should be reasonable with custom weights"


@pytest.mark.xfail(
    reason="cuML and sklearn don't have matching exceptions yet"
)
def test_knn_classifier_invalid_algorithm(classification_data):
    X, y = classification_data
    with pytest.raises((ValueError, KeyError)):
        model = KNeighborsClassifier(algorithm="invalid_algorithm")
        model.fit(X, y)


@pytest.mark.xfail(
    reason="cuML and sklearn don't have matching exceptions yet"
)
def test_knn_classifier_invalid_metric(classification_data):
    X, y = classification_data
    with pytest.raises(ValueError):
        model = KNeighborsClassifier(metric="invalid_metric")
        model.fit(X, y)


def test_knn_classifier_invalid_weights(classification_data):
    X, y = classification_data
    with pytest.raises(ValueError):
        model = KNeighborsClassifier(weights="invalid_weight")
        model.fit(X, y)


def test_knn_classifier_predict_proba(classification_data):
    X, y = classification_data
    model = KNeighborsClassifier()
    model.fit(X, y)
    proba = model.predict_proba(X)
    # Check that probabilities sum to 1
    assert np.allclose(proba.sum(axis=1), 1), "Probabilities should sum to 1"
    # Check shape
    assert proba.shape == (
        X.shape[0],
        len(np.unique(y)),
    ), "Probability matrix shape should be (n_samples, n_classes)"


def test_knn_classifier_sparse_input():
    from scipy.sparse import csr_matrix

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_sparse = csr_matrix(X)
    model = KNeighborsClassifier()
    model.fit(X_sparse, y)
    y_pred = model.predict(X_sparse)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7, "Accuracy should be reasonable with sparse input"


def test_knn_classifier_multilabel():
    from sklearn.datasets import make_multilabel_classification

    X, y = make_multilabel_classification(
        n_samples=100, n_features=20, n_classes=3, random_state=42
    )
    model = KNeighborsClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    # Check that the predicted shape matches the true labels
    assert (
        y_pred.shape == y.shape
    ), "Predicted labels should have the same shape as true labels"
    # Calculate accuracy for multi-label
    acc = (y_pred == y).mean()
    assert (
        acc > 0.7
    ), "Accuracy should be reasonable for multi-label classification"
