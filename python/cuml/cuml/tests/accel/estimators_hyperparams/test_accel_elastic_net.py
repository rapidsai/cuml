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
from sklearn.linear_model import ElasticNet
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


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 2.0])
def test_elasticnet_alpha(regression_data, alpha):
    X, y = regression_data
    model = ElasticNet(alpha=alpha, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for alpha={alpha}"


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 0.7, 1.0])
def test_elasticnet_l1_ratio(regression_data, l1_ratio):
    X, y = regression_data
    model = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for l1_ratio={l1_ratio}"
    # Check sparsity of coefficients when l1_ratio=1 (equivalent to Lasso)
    if l1_ratio == 1.0:
        num_nonzero = np.sum(model.coef_ != 0)
        assert (
            num_nonzero < X.shape[1]
        ), "Some coefficients should be zero when l1_ratio=1.0"


@pytest.mark.parametrize("max_iter", [100])
def test_elasticnet_max_iter(regression_data, max_iter):
    X, y = regression_data
    model = ElasticNet(max_iter=max_iter, random_state=42)
    model.fit(X, y)


@pytest.mark.parametrize("tol", [1e-3])
def test_elasticnet_tol(regression_data, tol):
    X, y = regression_data
    model = ElasticNet(tol=tol, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5, f"R^2 score should be reasonable for tol={tol}"


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_elasticnet_fit_intercept(regression_data, fit_intercept):
    X, y = regression_data
    model = ElasticNet(fit_intercept=fit_intercept, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with fit_intercept={fit_intercept}"


@pytest.mark.parametrize("precompute", [True, False])
def test_elasticnet_precompute(regression_data, precompute):
    X, y = regression_data
    model = ElasticNet(precompute=precompute, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with precompute={precompute}"


@pytest.mark.parametrize("selection", ["cyclic", "random"])
def test_elasticnet_selection(regression_data, selection):
    X, y = regression_data
    model = ElasticNet(selection=selection, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    # Compute R^2 score
    r2 = r2_score(y, y_pred)
    assert (
        r2 > 0.5
    ), f"R^2 score should be reasonable with selection={selection}"


def test_elasticnet_random_state(regression_data):
    X, y = regression_data
    model1 = ElasticNet(selection="random", random_state=42)
    model1.fit(X, y)
    model2 = ElasticNet(selection="random", random_state=42)
    model2.fit(X, y)
    # Coefficients should be the same when random_state is fixed
    np.testing.assert_allclose(
        model1.coef_,
        model2.coef_,
        err_msg="Coefficients should be the same with the same random_state",
    )
    model3 = ElasticNet(selection="random", random_state=24)
    model3.fit(X, y)


@pytest.mark.xfail(reason="cuML does not emit ConvergenceWarning yet.")
def test_elasticnet_convergence_warning(regression_data):
    X, y = regression_data
    from sklearn.exceptions import ConvergenceWarning

    with pytest.warns(ConvergenceWarning):
        model = ElasticNet(max_iter=1, random_state=42)
        model.fit(X, y)


def test_elasticnet_coefficients(regression_data):
    X, y = regression_data
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    model.fit(X, y)
    coef_nonzero = np.sum(model.coef_ != 0)
    assert coef_nonzero > 0, "There should be non-zero coefficients"


def test_elasticnet_l1_ratio_effect(regression_data):
    X, y = regression_data
    model_l1 = ElasticNet(alpha=0.1, l1_ratio=1.0, random_state=42)
    model_l1.fit(X, y)
    model_l2 = ElasticNet(alpha=0.1, l1_ratio=0.0, random_state=42)
    model_l2.fit(X, y)
    num_nonzero_l1 = np.sum(model_l1.coef_ != 0)
    num_nonzero_l2 = np.sum(model_l2.coef_ != 0)
    assert (
        num_nonzero_l1 <= num_nonzero_l2
    ), "L1 regularization should produce sparser coefficients than L2"


@pytest.mark.parametrize("copy_X", [True, False])
def test_elasticnet_copy_X(regression_data, copy_X):
    X, y = regression_data
    X_original = X.copy()
    model = ElasticNet(copy_X=copy_X, random_state=42)
    model.fit(X, y)
    if copy_X:
        # X should remain unchanged
        assert np.allclose(
            X, X_original
        ), "X has been modified when copy_X=True"
    else:
        # X might be modified when copy_X=False
        pass  # We cannot guarantee X remains unchanged


def test_elasticnet_positive(regression_data):
    X, y = regression_data
    model = ElasticNet(positive=True, random_state=42)
    model.fit(X, y)
    # All coefficients should be non-negative
    assert np.all(
        model.coef_ >= 0
    ), "All coefficients should be non-negative when positive=True"


def test_elasticnet_warm_start(regression_data):
    X, y = regression_data
    model = ElasticNet(warm_start=True, random_state=42)
    model.fit(X, y)
    coef_old = model.coef_.copy()
    # Fit again with more iterations
    model.set_params(max_iter=2000)
    model.fit(X, y)
    coef_new = model.coef_
    # Coefficients should change after more iterations
    assert not np.allclose(
        coef_old, coef_new
    ), "Coefficients should update when warm_start=True"
