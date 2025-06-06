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

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=100, n_features=20, noise=0.1, random_state=42
    )
    return X, y


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_regression_fit_intercept(regression_data, fit_intercept):
    X, y = regression_data
    lr = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    lr.predict(X)


@pytest.mark.parametrize("copy_X", [True, False])
def test_linear_regression_copy_X(regression_data, copy_X):
    X, y = regression_data
    X_original = X.copy()
    LinearRegression(copy_X=copy_X).fit(X, y)
    if copy_X:
        # X should remain unchanged
        assert np.array_equal(
            X, X_original
        ), "X has been modified when copy_X=True"
    else:
        # X might be modified when copy_X=False
        pass  # We cannot guarantee X remains unchanged


@pytest.mark.parametrize("positive", [True, False])
def test_linear_regression_positive(regression_data, positive):
    X, y = regression_data
    lr = LinearRegression(positive=positive).fit(X, y)
    lr.predict(X)
    if positive:
        # Verify that all coefficients are non-negative
        assert np.all(lr.coef_ >= 0), "Not all coefficients are non-negative"
