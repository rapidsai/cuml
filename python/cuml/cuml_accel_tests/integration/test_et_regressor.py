# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=300, n_features=20, n_informative=10, random_state=42,
    )
    return X, y


@pytest.mark.parametrize("n_estimators", [10, 50])
def test_et_n_estimators(regression_data, n_estimators):
    X, y = regression_data
    reg = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_et_criterion(regression_data, criterion):
    X, y = regression_data
    reg = ExtraTreesRegressor(
        criterion=criterion, n_estimators=20, random_state=42,
    )
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


@pytest.mark.parametrize("max_depth", [None, 5, 10])
def test_et_max_depth(regression_data, max_depth):
    X, y = regression_data
    reg = ExtraTreesRegressor(
        max_depth=max_depth, n_estimators=20, random_state=42,
    )
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


def test_et_bootstrap_oob_score(regression_data):
    X, y = regression_data
    reg = ExtraTreesRegressor(
        bootstrap=True, oob_score=True, n_estimators=20, random_state=42,
    )
    reg.fit(X, y)
    assert hasattr(reg, "oob_score_")
