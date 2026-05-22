# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=300, n_features=20, n_informative=10, n_redundant=5,
        random_state=42,
    )
    return X, y


@pytest.mark.parametrize("n_estimators", [10, 50])
def test_et_n_estimators(classification_data, n_estimators):
    X, y = classification_data
    clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.5


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_et_criterion(classification_data, criterion):
    X, y = classification_data
    clf = ExtraTreesClassifier(
        criterion=criterion, n_estimators=20, random_state=42,
    )
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.5


@pytest.mark.parametrize("max_depth", [None, 5, 10])
def test_et_max_depth(classification_data, max_depth):
    X, y = classification_data
    clf = ExtraTreesClassifier(
        max_depth=max_depth, n_estimators=20, random_state=42,
    )
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.5


@pytest.mark.parametrize("class_weight", [None, "balanced", {0: 1, 1: 2}])
def test_et_class_weight(classification_data, class_weight):
    X, y = classification_data
    clf = ExtraTreesClassifier(
        class_weight=class_weight, n_estimators=20, random_state=42,
    )
    clf.fit(X, y)
    assert accuracy_score(y, clf.predict(X)) > 0.5


def test_et_predict_proba(classification_data):
    X, y = classification_data
    clf = ExtraTreesClassifier(n_estimators=20, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)


def test_et_bootstrap_oob_score(classification_data):
    X, y = classification_data
    clf = ExtraTreesClassifier(
        bootstrap=True, oob_score=True, n_estimators=20, random_state=42,
    )
    clf.fit(X, y)
    assert hasattr(clf, "oob_score_")
