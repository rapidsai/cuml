#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def classification_data():
    # Create a synthetic classification dataset.
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    return X, y


@pytest.mark.parametrize("n_estimators", [10, 50, 100])
def test_rf_n_estimators(classification_data, n_estimators):
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_rf_criterion(classification_data, criterion):
    X, y = classification_data
    clf = RandomForestClassifier(
        criterion=criterion, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("max_depth", [None, 5, 10])
def test_rf_max_depth(classification_data, max_depth):
    X, y = classification_data
    clf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("min_samples_split", [2, 5, 10])
def test_rf_min_samples_split(classification_data, min_samples_split):
    X, y = classification_data
    clf = RandomForestClassifier(
        min_samples_split=min_samples_split, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("min_samples_leaf", [1, 2, 4])
def test_rf_min_samples_leaf(classification_data, min_samples_leaf):
    X, y = classification_data
    clf = RandomForestClassifier(
        min_samples_leaf=min_samples_leaf, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("min_weight_fraction_leaf", [0.0, 0.1])
def test_rf_min_weight_fraction_leaf(
    classification_data, min_weight_fraction_leaf
):
    X, y = classification_data
    clf = RandomForestClassifier(
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        n_estimators=50,
        random_state=42,
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("max_features", ["sqrt", "log2", 0.5, 5])
def test_rf_max_features(classification_data, max_features):
    X, y = classification_data
    clf = RandomForestClassifier(
        max_features=max_features, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("max_leaf_nodes", [None, 10, 20])
def test_rf_max_leaf_nodes(classification_data, max_leaf_nodes):
    X, y = classification_data
    clf = RandomForestClassifier(
        max_leaf_nodes=max_leaf_nodes, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("min_impurity_decrease", [0.0, 0.1])
def test_rf_min_impurity_decrease(classification_data, min_impurity_decrease):
    X, y = classification_data
    clf = RandomForestClassifier(
        min_impurity_decrease=min_impurity_decrease,
        n_estimators=50,
        random_state=42,
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("bootstrap", [True, False])
def test_rf_bootstrap(classification_data, bootstrap):
    X, y = classification_data
    clf = RandomForestClassifier(
        bootstrap=bootstrap, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_rf_n_jobs(classification_data, n_jobs):
    X, y = classification_data
    clf = RandomForestClassifier(
        n_jobs=n_jobs, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("verbose", [0, 1])
def test_rf_verbose(classification_data, verbose):
    X, y = classification_data
    clf = RandomForestClassifier(
        verbose=verbose, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("warm_start", [False, True])
def test_rf_warm_start(classification_data, warm_start):
    X, y = classification_data
    clf = RandomForestClassifier(
        warm_start=warm_start, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("class_weight", [None, "balanced", {0: 1, 1: 2}])
def test_rf_class_weight(classification_data, class_weight):
    X, y = classification_data
    clf = RandomForestClassifier(
        class_weight=class_weight, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("ccp_alpha", [0.0, 0.1])
def test_rf_ccp_alpha(classification_data, ccp_alpha):
    X, y = classification_data
    clf = RandomForestClassifier(
        ccp_alpha=ccp_alpha, n_estimators=50, random_state=42
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


@pytest.mark.parametrize("max_samples", [None, 0.8, 50])
def test_rf_max_samples(classification_data, max_samples):
    X, y = classification_data
    clf = RandomForestClassifier(
        max_samples=max_samples,
        bootstrap=True,
        n_estimators=50,
        random_state=42,
    )
    clf.fit(X, y)
    _ = accuracy_score(y, clf.predict(X))


def test_rf_random_state(classification_data):
    X, y = classification_data
    clf1 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    # Predictions should be identical with the same random_state.
    assert np.array_equal(clf1.predict(X), clf2.predict(X))
