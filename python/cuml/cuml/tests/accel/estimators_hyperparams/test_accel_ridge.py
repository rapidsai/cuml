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


import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    # Standardize features
    X = StandardScaler().fit_transform(X)
    return X, y


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
def test_ridge_alpha(regression_data, alpha):
    X, y = regression_data
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for alpha={alpha}"


@pytest.mark.parametrize(
    "solver",
    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
)
def test_ridge_solver(regression_data, solver):
    X, y = regression_data
    positive = solver == "lbfgs"
    model = Ridge(solver=solver, random_state=42, positive=positive)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable with solver={solver}"


@pytest.mark.parametrize("max_iter", [100])
def test_ridge_max_iter(regression_data, max_iter):
    X, y = regression_data
    model = Ridge(max_iter=max_iter, solver="sag", random_state=42)
    model.fit(X, y)


@pytest.mark.parametrize("tol", [1e-4, 1e-3, 1e-2])
def test_ridge_tol(regression_data, tol):
    X, y = regression_data
    model = Ridge(tol=tol, solver="sag", random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for tol={tol}"


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_fit_intercept(regression_data, fit_intercept):
    X, y = regression_data
    model = Ridge(fit_intercept=fit_intercept, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with fit_intercept={fit_intercept}"


def test_ridge_random_state(regression_data):
    X, y = regression_data
    model1 = Ridge(solver="sag", random_state=42)
    model1.fit(X, y)
    model2 = Ridge(solver="sag", random_state=42)
    model2.fit(X, y)
    # Coefficients should be the same when random_state is fixed
    np.testing.assert_allclose(
        model1.coef_,
        model2.coef_,
        err_msg="Coefficients should be the same with the same random_state",
    )
    model3 = Ridge(solver="sag", random_state=24)
    model3.fit(X, y)


@pytest.mark.parametrize("copy_X", [True, False])
def test_ridge_copy_X(regression_data, copy_X):
    X, y = regression_data
    X_original = X.copy()
    model = Ridge(copy_X=copy_X, random_state=42)
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
def test_ridge_convergence_warning(regression_data):
    X, y = regression_data
    from sklearn.exceptions import ConvergenceWarning

    with pytest.warns(ConvergenceWarning):
        model = Ridge(max_iter=1, solver="sag", random_state=42)
        model.fit(X, y)


def test_ridge_coefficients(regression_data):
    X, y = regression_data
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    coef_nonzero = np.sum(model.coef_ != 0)
    assert coef_nonzero > 0, "There should be non-zero coefficients"


def test_ridge_positive(regression_data):
    X, y = regression_data
    model = Ridge(positive=True, solver="lbfgs", random_state=42)
    model.fit(X, y)
    # All coefficients should be non-negative
    assert np.all(
        model.coef_ >= 0
    ), "All coefficients should be non-negative when positive=True"
