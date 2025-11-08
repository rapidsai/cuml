# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=2,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture(scope="module")
def multiclass():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


def test_svc_linear(binary):
    X, y = binary
    svc = LinearSVC().fit(X, y)
    assert svc.score(X, y) > 0.5


def test_svc_linear_multiclass(multiclass):
    X, y = multiclass
    svr = LinearSVC().fit(X, y)
    assert svr.score(X, y) > 0.5
