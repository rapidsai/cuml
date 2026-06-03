# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import scipy.sparse
import sklearn.linear_model
from hypothesis import example, given
from hypothesis import strategies as st
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import train_test_split

import cuml
from cuml.metrics import r2_score
from cuml.testing.datasets import make_regression_dataset
from cuml.testing.strategies import dataset_dtypes
from cuml.testing.utils import quality_param, stress_param, unit_param


@given(
    datatype=dataset_dtypes(),
    alpha=st.sampled_from([0.1, 1.0, 10.0]),
    l1_ratio=st.sampled_from([0.1, 0.5, 0.9]),
    nrows=st.integers(min_value=1000, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
)
@example(
    datatype=np.float32,
    alpha=0.1,
    l1_ratio=0.1,
    nrows=1000,
    column_info=[20, 10],
)
@example(
    datatype=np.float64,
    alpha=10.0,
    l1_ratio=0.9,
    nrows=5000,
    column_info=[100, 50],
)
def test_elastic_net_solvers_eq(datatype, alpha, l1_ratio, nrows, column_info):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    kwargs = {"alpha": alpha, "l1_ratio": l1_ratio}
    cd = cuml.ElasticNet(solver="cd", **kwargs)
    cd.fit(X_train, y_train)
    cd_res = cd.predict(X_test)

    qn = cuml.ElasticNet(solver="qn", **kwargs)
    qn.fit(X_train, y_train)
    # the results of the two models should be close (even if both are bad)
    assert qn.score(X_test, cd_res) > 0.90
    # coefficients of the two models should be close
    assert np.corrcoef(cd.coef_, qn.coef_)[0, 1] > 0.98


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.1, 0.001])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.filterwarnings("ignore:Objective did not converge::sklearn[.*]")
def test_lasso(datatype, alpha, algorithm, nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    cu_lasso = cuml.Lasso(
        alpha=alpha,
        fit_intercept=True,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
    )

    cu_lasso.fit(X_train, y_train)
    assert cu_lasso.coef_ is not None
    cu_predict = cu_lasso.predict(X_test)

    cu_r2 = r2_score(y_test, cu_predict)

    if nrows < 500000:
        sk_lasso = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=1000,
            selection=algorithm,
            tol=1e-10,
        )
        sk_lasso.fit(X_train, y_train)
        sk_predict = sk_lasso.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict)
        assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
def test_lasso_default(datatype, nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    cu_lasso = cuml.Lasso()

    cu_lasso.fit(X_train, y_train)
    assert cu_lasso.coef_ is not None
    cu_predict = cu_lasso.predict(X_test)
    cu_r2 = r2_score(y_test, cu_predict)

    sk_lasso = Lasso()
    sk_lasso.fit(X_train, y_train)
    sk_predict = sk_lasso.predict(X_test)
    sk_r2 = r2_score(y_test, sk_predict)
    assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("model", ["lasso", "elastic-net"])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "distribution", ["lognormal", "exponential", "uniform"]
)
@pytest.mark.filterwarnings("ignore:Objective did not converge::sklearn[.*]")
def test_weighted_cd(datatype, model, fit_intercept, distribution):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20
    X, y = make_regression(nrows, ncols, n_informative=n_info, noise=noise)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
    elif distribution == "exponential":
        wt = np.random.exponential(scale=max_weight, size=len(X_train))
    else:
        wt = np.random.lognormal(size=len(X_train))

    cuModel = cuml.Lasso if model == "lasso" else cuml.ElasticNet
    skModel = Lasso if model == "lasso" else ElasticNet

    # Initialization of cuML's linear regression model
    cumodel = cuModel(fit_intercept=fit_intercept, tol=1e-10, max_iter=1000)

    # fit and predict cuml linear regression model
    cumodel.fit(X_train, y_train, sample_weight=wt)
    cumodel_predict = cumodel.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skmodel = skModel(fit_intercept=fit_intercept, tol=1e-10, max_iter=1000)
    skmodel.fit(X_train, y_train, sample_weight=wt)

    skmodel_predict = skmodel.predict(X_test)
    cu_r2 = r2_score(y_test, cumodel_predict)
    sk_r2 = r2_score(y_test, skmodel_predict)
    assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.2, 0.7])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.filterwarnings("ignore:Objective did not converge::sklearn[.*]")
def test_elastic_net(datatype, alpha, algorithm, nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    elastic_cu = cuml.ElasticNet(
        alpha=alpha,
        fit_intercept=True,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
    )

    elastic_cu.fit(X_train, y_train)
    cu_predict = elastic_cu.predict(X_test)

    cu_r2 = r2_score(y_test, cu_predict)

    if nrows < 500000:
        elastic_sk = ElasticNet(
            alpha=alpha,
            fit_intercept=True,
            max_iter=1000,
            selection=algorithm,
            tol=1e-10,
        )
        elastic_sk.fit(X_train, y_train)
        sk_predict = elastic_sk.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict)

        assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
def test_elastic_net_default(datatype, nrows, column_info):
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    elastic_cu = cuml.ElasticNet()
    elastic_cu.fit(X_train, y_train)
    cu_predict = elastic_cu.predict(X_test)
    cu_r2 = r2_score(y_test, cu_predict)

    elastic_sk = ElasticNet()
    elastic_sk.fit(X_train, y_train)
    sk_predict = elastic_sk.predict(X_test)
    sk_r2 = r2_score(y_test, sk_predict)
    assert cu_r2 >= sk_r2 - 0.07


@pytest.mark.parametrize("train_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_dtype", [np.float64, np.float32])
def test_elastic_net_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=0
    )
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    clf = cuml.ElasticNet()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize("train_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_dtype", [np.float64, np.float32])
def test_lasso_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=0
    )
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    clf = cuml.Lasso()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize("cls", [cuml.ElasticNet, cuml.Lasso])
def test_set_params(cls):
    x = np.linspace(0, 1, 50)[:, None]
    y = 2 * x

    model = cls(alpha=0.01)
    model.fit(x, y)
    coef_before = model.coef_

    model = cls(selection="random", alpha=0.1)
    model.fit(x, y)
    coef_after = model.coef_

    model = cls(alpha=0.01)
    model.set_params(**{"selection": "random", "alpha": 0.1})
    model.fit(x, y)
    coef_test = model.coef_

    assert coef_before != coef_after
    assert coef_after == coef_test


@pytest.mark.parametrize("cls", [cuml.ElasticNet, cuml.Lasso])
@pytest.mark.parametrize("solver", ["qn", "cd"])
def test_max_iter_n_iter(cls, solver):
    X, y = make_regression(random_state=42)

    model = cls(max_iter=2, solver=solver).fit(X, y)
    assert model.n_iter_ == 2


def make_sparse_regression(
    n_samples=1000, n_features=100, n_informative=10, seed=42, dtype="float64"
):
    rng = np.random.default_rng(seed)

    w = rng.normal(size=(n_features, 1))
    w[n_informative:] = 0.0

    X = rng.normal(size=(n_samples, n_features))
    rnd = rng.uniform(size=(n_samples, n_features))
    X[rnd > 0.5] = 0.0

    y = np.dot(X, w).ravel().astype(dtype)
    X = scipy.sparse.csr_matrix(X).astype(dtype)
    return X, y


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("alpha", [0.2, 0.7])
@pytest.mark.parametrize("model_name", ["ElasticNet", "Lasso"])
def test_sparse(dtype, alpha, model_name):
    X, y = make_sparse_regression(dtype=dtype)

    cu_cls = getattr(cuml.linear_model, model_name)
    sk_cls = getattr(sklearn.linear_model, model_name)

    cu_model = cu_cls(alpha=alpha, tol=1e-10).fit(X, y)
    sk_model = sk_cls(alpha=alpha).fit(X, y)

    np.testing.assert_allclose(cu_model.coef_, sk_model.coef_, atol=1e-3)
    np.testing.assert_allclose(
        cu_model.intercept_, sk_model.intercept_, atol=1e-3
    )

    assert isinstance(cu_model.sparse_coef_, scipy.sparse.csr_matrix)

    cu_score = cu_model.score(X, y)
    sk_score = sk_model.score(X, y)
    assert cu_score >= sk_score - 0.1


def test_solver_errors():
    X, y = make_sparse_regression()

    with pytest.raises(ValueError, match="solver='bad' is not supported"):
        cuml.ElasticNet(solver="bad").fit(X, y)

    with pytest.raises(ValueError, match="solver='cd' doesn't support sparse"):
        cuml.ElasticNet(solver="cd").fit(X, y)
