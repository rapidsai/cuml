# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


@pytest.fixture(scope="module")
def linear_X_y():
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


def test_svr_linear(linear_X_y):
    X, y = linear_X_y
    svr = LinearSVR().fit(X, y)
    assert svr.score(X, y) > 0.5
