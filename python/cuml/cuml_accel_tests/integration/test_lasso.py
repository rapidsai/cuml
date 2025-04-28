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

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def regression_data():
    X, y, coef = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=0.1,
        coef=True,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y, coef


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
def test_lasso_alpha(regression_data, alpha):
    X, y, _ = regression_data
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2_score(y, y_pred)


def test_lasso_alpha_sparsity(regression_data):
    X, y, _ = regression_data
    alphas = [0.1, 1.0, 10.0, 100.0]
    zero_counts = []
    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X, y)
        zero_counts.append(np.sum(model.coef_ == 0))
    # Check that zero_counts increases with alpha
    assert zero_counts == sorted(
        zero_counts
    ), "Number of zero coefficients should increase with alpha"


@pytest.mark.parametrize("max_iter", [100])
def test_lasso_max_iter(regression_data, max_iter):
    X, y, _ = regression_data
    model = Lasso(max_iter=max_iter, random_state=42)
    model.fit(X, y)


@pytest.mark.parametrize("tol", [1e-3])
def test_lasso_tol(regression_data, tol):
    X, y, _ = regression_data
    model = Lasso(tol=tol, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for tol={tol}"


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_lasso_fit_intercept(regression_data, fit_intercept):
    X, y, _ = regression_data
    model = Lasso(fit_intercept=fit_intercept, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with fit_intercept={fit_intercept}"


def test_lasso_positive(regression_data):
    X, y, _ = regression_data
    model = Lasso(positive=True, random_state=42)
    model.fit(X, y)
    # All coefficients should be non-negative
    assert np.all(
        model.coef_ >= 0
    ), "All coefficients should be non-negative when positive=True"


def test_lasso_random_state(regression_data):
    X, y, _ = regression_data
    model1 = Lasso(selection="random", random_state=42)
    model1.fit(X, y)
    model2 = Lasso(selection="random", random_state=42)
    model2.fit(X, y)
    # Coefficients should be the same when random_state is fixed
    np.testing.assert_allclose(
        model1.coef_,
        model2.coef_,
        err_msg="Coefficients should be the same with the same random_state",
    )
    model3 = Lasso(selection="random", random_state=24)
    model3.fit(X, y)


def test_lasso_warm_start(regression_data):
    X, y, _ = regression_data
    model = Lasso(warm_start=True, random_state=42)
    model.fit(X, y)
    coef_old = model.coef_.copy()
    # Fit again with different alpha
    model.set_params(alpha=10.0)
    model.fit(X, y)
    coef_new = model.coef_
    # Coefficients should change after refitting with a different alpha
    assert not np.allclose(
        coef_old, coef_new
    ), "Coefficients should update when warm_start=True"


@pytest.mark.parametrize("copy_X", [True, False])
def test_lasso_copy_X(regression_data, copy_X):
    X, y, _ = regression_data
    X_original = X.copy()
    model = Lasso(copy_X=copy_X, random_state=42)
    model.fit(X, y)
    if copy_X:
        # X should remain unchanged
        assert np.allclose(
            X, X_original
        ), "X has been modified when copy_X=True"
    else:
        # X might be modified when copy_X=False
        pass  # We cannot guarantee X remains unchanged


@pytest.mark.xfail(reason="cuML does not emit ConvergenceWarning yet.")
def test_lasso_convergence_warning(regression_data):
    X, y, _ = regression_data
    from sklearn.exceptions import ConvergenceWarning

    with pytest.warns(ConvergenceWarning):
        model = Lasso(max_iter=1, random_state=42)
        model.fit(X, y)


def test_lasso_coefficients_sparsity(regression_data):
    X, y, _ = regression_data
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X, y)
    coef_zero = np.sum(model.coef_ == 0)
    assert (
        coef_zero > 0
    ), "There should be zero coefficients indicating sparsity"


@pytest.mark.parametrize("selection", ["cyclic", "random"])
def test_lasso_selection(regression_data, selection):
    X, y, _ = regression_data
    model = Lasso(selection=selection, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with selection={selection}"


@pytest.mark.parametrize("precompute", [True, False])
def test_lasso_precompute(regression_data, precompute):
    X, y, _ = regression_data
    model = Lasso(precompute=precompute, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with precompute={precompute}"
