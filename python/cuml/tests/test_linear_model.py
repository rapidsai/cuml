# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from functools import lru_cache
import cupy as cp
import numpy as np
import pytest
from hypothesis import (
    assume,
    example,
    given,
    settings,
    strategies as st,
    target
)
from hypothesis.extra.numpy import floating_dtypes
from distutils.version import LooseVersion
import cudf
from cuml import ElasticNet as cuElasticNet
from cuml import LinearRegression as cuLinearRegression
from cuml import LogisticRegression as cuLog
from cuml import Ridge as cuRidge
from cuml.internals.array import elements_in_representable_range
from cuml.testing.strategies import (
    regression_datasets,
    split_datasets,
    standard_regression_datasets,
)
from cuml.testing.utils import (
    array_difference,
    array_equal,
    small_regression_dataset,
    small_classification_dataset,
    unit_param,
    quality_param,
    stress_param,
)
import rmm

from scipy.sparse import csr_matrix

import sklearn
from sklearn.datasets import make_regression, make_classification, load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import LogisticRegression as skLog
from sklearn.model_selection import train_test_split


pytestmark = pytest.mark.filterwarnings("ignore: Regressors in active "
                                        "set degenerate(.*)::sklearn[.*]")


def _make_regression_dataset_uncached(nrows, ncols, n_info, **kwargs):
    X, y = make_regression(
        **kwargs, n_samples=nrows, n_features=ncols, n_informative=n_info,
        random_state=0
    )
    return train_test_split(X, y, train_size=0.8, random_state=10)


@lru_cache(4)
def _make_regression_dataset_from_cache(nrows, ncols, n_info, **kwargs):
    return _make_regression_dataset_uncached(nrows, ncols, n_info, **kwargs)


def make_regression_dataset(datatype, nrows, ncols, n_info, **kwargs):
    if nrows * ncols < 1e8:  # Keep cache under 4 GB
        dataset = _make_regression_dataset_from_cache(nrows, ncols, n_info,
                                                      **kwargs)
    else:
        dataset = _make_regression_dataset_uncached(nrows, ncols, n_info,
                                                    **kwargs)

    return map(lambda arr: arr.astype(datatype), dataset)


def make_classification_dataset(datatype, nrows, ncols, n_info, num_classes):
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_classes=num_classes,
        random_state=0,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=10
    )

    return X_train, X_test, y_train, y_test


def sklearn_compatible_dataset(X_train, X_test, y_train, _=None):
    return (
        X_train.shape[1] >= 1
        and (X_train > 0).any()
        and (y_train > 0).any()
        and all(np.isfinite(x).all()
                for x in (X_train, X_test, y_train) if x is not None)
    )


def cuml_compatible_dataset(X_train, X_test, y_train, _=None):
    return (
        X_train.shape[0] >= 2
        and X_train.shape[1] >= 1
        and np.isfinite(X_train).all()
        and all(
            elements_in_representable_range(x, np.float32)
            for x in (X_train, X_test, y_train)
            if x is not None
        )
    )


@pytest.mark.parametrize("ntargets", [1, 2])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["eig", "svd"])
@pytest.mark.parametrize(
    "nrows", [unit_param(1000), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500])
    ],
)
def test_linear_regression_model(
    datatype, algorithm, nrows, column_info, ntargets
):
    if algorithm == "svd" and nrows > 46340:
        pytest.skip("svd solver is not supported for the data that has more"
                    "than 46340 rows or columns if you are using CUDA version"
                    "10.x")
    if 1 < ntargets and algorithm != "svd":
        pytest.skip("The multi-target fit only supports using the svd solver.")

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, n_targets=ntargets
    )

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    if nrows < 500000:
        # sklearn linear regression model initialization, fit and predict
        skols = skLinearRegression(fit_intercept=True, normalize=False)
        skols.fit(X_train, y_train)

        skols_predict = skols.predict(X_test)

        assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@pytest.mark.parametrize("ntargets", [1, 2])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["eig", "svd", "qr", "svd-qr"])
@pytest.mark.parametrize(
    "fit_intercept, normalize, distribution", [
        (True, True, "lognormal"),
        (True, True, "exponential"),
        (True, False, "uniform"),
        (True, False, "exponential"),
        (False, True, "lognormal"),
        (False, False, "uniform"),
    ]
)
def test_weighted_linear_regression(
    ntargets, datatype, algorithm, fit_intercept, normalize, distribution
):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20

    if 1 < ntargets and normalize:
        pytest.skip("The multi-target fit does not support normalization.")
    if 1 < ntargets and algorithm != "svd":
        pytest.skip("The multi-target fit only supports using the svd solver.")

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, noise=noise, n_targets=ntargets
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
    elif distribution == "exponential":
        wt = np.random.exponential(size=len(X_train))
    else:
        wt = np.random.lognormal(size=len(X_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=fit_intercept,
                               normalize=normalize,
                               algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train, sample_weight=wt)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skols = skLinearRegression(fit_intercept=fit_intercept,
                               normalize=normalize)
    skols.fit(X_train, y_train, sample_weight=wt)

    skols_predict = skols.predict(X_test)

    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@pytest.mark.skipif(
    rmm._cuda.gpu.runtimeGetVersion() < 11000,
    reason='svd solver does not support more than 46340 rows or columns for'
           ' CUDA<11 and other solvers do not support single-column input'
)
def test_linear_regression_single_column():
    '''Test that linear regression can be run on single column with more than
    46340 rows (a limitation on CUDA <11)'''
    model = cuLinearRegression()
    with pytest.warns(UserWarning):
        model.fit(cp.random.rand(46341), cp.random.rand(46341))


# The assumptions required to have this test pass are relatively strong.
# It should be possible to relax assumptions once #4963 is resolved.
# See also: test_linear_regression_model_default_generalized
@given(
    split_datasets(
        standard_regression_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_samples=st.just(1000),
        ),
        test_sizes=st.just(0.2)
    )
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
@settings(deadline=5000)
def test_linear_regression_model_default(dataset):

    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    target(float(array_difference(skols_predict, cuols_predict)))
    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


# TODO: Replace test_linear_regression_model_default with this test once #4963
# is resolved.
@pytest.mark.skip(reason="https://github.com/rapidsai/cuml/issues/4963")
@given(
    split_datasets(regression_datasets(dtypes=floating_dtypes(sizes=(32, 64))))
)
@settings(deadline=5000)
def test_linear_regression_model_default_generalized(dataset):

    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression()

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization and fit
    skols = skLinearRegression()
    skols.fit(X_train, y_train)

    skols_predict = skols.predict(X_test)

    target(float(array_difference(skols_predict, cuols_predict)))
    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_ridge_regression_model_default(datatype):

    X_train, X_test, y_train, y_test = small_regression_dataset(datatype)

    curidge = cuRidge()

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    # sklearn ridge regression model initialization, fit and predict
    skridge = skRidge()
    skridge.fit(X_train, y_train)
    skridge_predict = skridge.predict(X_test)

    assert array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["eig", "svd"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500])
    ],
)
def test_ridge_regression_model(datatype, algorithm, nrows, column_info):

    if algorithm == "svd" and nrows > 46340:
        pytest.skip("svd solver is not supported for the data that has more"
                    "than 46340 rows or columns if you are using CUDA version"
                    "10.x")

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    # Initialization of cuML's ridge regression model
    curidge = cuRidge(fit_intercept=False, normalize=False, solver=algorithm)

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    if nrows < 500000:
        # sklearn ridge regression model initialization, fit and predict
        skridge = skRidge(fit_intercept=False, normalize=False)
        skridge.fit(X_train, y_train)

        skridge_predict = skridge.predict(X_test)

        assert array_equal(skridge_predict,
                           curidge_predict,
                           1e-1,
                           with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["eig", "svd"])
@pytest.mark.parametrize(
    "fit_intercept, normalize, distribution", [
        (True, True, "lognormal"),
        (True, True, "exponential"),
        (True, False, "uniform"),
        (True, False, "exponential"),
        (False, True, "lognormal"),
        (False, False, "uniform"),
    ]
)
def test_weighted_ridge(datatype, algorithm, fit_intercept,
                        normalize, distribution):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, noise=noise
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
    elif distribution == "exponential":
        wt = np.random.exponential(size=len(X_train))
    else:
        wt = np.random.lognormal(size=len(X_train))

    # Initialization of cuML's linear regression model
    curidge = cuRidge(fit_intercept=fit_intercept,
                      normalize=normalize,
                      solver=algorithm)

    # fit and predict cuml linear regression model
    curidge.fit(X_train, y_train, sample_weight=wt)
    curidge_predict = curidge.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skridge = skRidge(fit_intercept=fit_intercept,
                      normalize=normalize)
    skridge.fit(X_train, y_train, sample_weight=wt)

    skridge_predict = skridge.predict(X_test)

    assert array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)


@pytest.mark.parametrize(
    "num_classes, dtype, penalty, l1_ratio, fit_intercept, C, tol", [
        # L-BFGS Solver
        (2, np.float32, "none", 1.0, True, 1.0, 1e-3),
        (2, np.float64, "l2", 1.0, True, 1.0, 1e-8),
        (10, np.float32, "elasticnet", 0.0, True, 1.0, 1e-3),
        (10, np.float32, "none", 1.0, False, 1.0, 1e-8),
        (10, np.float32, "none", 1.0, False, 2.0, 1e-3),
        # OWL-QN Solver
        (2, np.float32, "l1", 1.0, True, 1.0, 1e-3),
        (2, np.float64, "elasticnet", 1.0, True, 1.0, 1e-8),
        (10, np.float32, "l1", 1.0, True, 1.0, 1e-3),
        (10, np.float32, "l1", 1.0, False, 1.0, 1e-8),
        (10, np.float32, "elasticnet", 1.0, False, 0.5, 1e-3),
    ]
)
@pytest.mark.parametrize("nrows", [unit_param(1000)])
@pytest.mark.parametrize("column_info", [unit_param([20, 10])])
# ignoring UserWarnings in sklearn about setting unused parameters
# like l1 for none penalty
@pytest.mark.filterwarnings("ignore::UserWarning:sklearn[.*]")
def test_logistic_regression(
    num_classes, dtype, penalty, l1_ratio,
    fit_intercept, nrows, column_info, C, tol
):
    ncols, n_info = column_info
    # Checking sklearn >= 0.21 for testing elasticnet
    sk_check = LooseVersion(str(sklearn.__version__)) >= LooseVersion("0.21.0")
    if not sk_check and penalty == "elasticnet":
        pytest.skip(
            "Need sklearn > 0.21 for testing logistic with" "elastic net."
        )

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype, nrows=nrows, ncols=ncols,
        n_info=n_info, num_classes=num_classes
    )
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    culog = cuLog(
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        fit_intercept=fit_intercept,
        tol=tol
    )
    culog.fit(X_train, y_train)

    # Only solver=saga supports elasticnet in scikit
    if penalty in ["elasticnet", "l1"]:
        if sk_check:
            sklog = skLog(
                penalty=penalty,
                l1_ratio=l1_ratio,
                solver="saga",
                C=C,
                fit_intercept=fit_intercept,
                multi_class="auto",
            )
        else:
            sklog = skLog(
                penalty=penalty,
                solver="saga",
                C=C,
                fit_intercept=fit_intercept,
                multi_class="auto",
            )
    else:
        sklog = skLog(
            penalty=penalty,
            solver="lbfgs",
            C=C,
            fit_intercept=fit_intercept,
            multi_class="auto",
        )

    sklog.fit(X_train, y_train)

    # Setting tolerance to lowest possible per loss to detect regressions
    # as much as possible
    cu_preds = culog.predict(X_test)
    tol_test = 0.012
    tol_train = 0.006
    if num_classes == 10 and penalty in ["elasticnet", "l1"]:
        tol_test *= 10
        tol_train *= 10

    assert culog.score(X_train, y_train) >= sklog.score(X_train, y_train) - \
        tol_train
    assert culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - \
        tol_test
    assert len(np.unique(cu_preds)) == len(np.unique(y_test))


@pytest.mark.parametrize("dtype, penalty, l1_ratio", [
    (np.float32, "none", 1.0),
    (np.float64, "l2", 0.0),
    (np.float32, "elasticnet", 1.0),
    (np.float64, "l1", None),
])
def test_logistic_regression_unscaled(dtype, penalty, l1_ratio):
    # Test logistic regression on the breast cancer dataset. We do not scale
    # the dataset which could lead to numerical problems (fixed in PR #2543).
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    params = {"penalty": penalty, "C": 1, "tol": 1e-4, "fit_intercept": True,
              'max_iter': 5000, "l1_ratio": l1_ratio}
    culog = cuLog(**params)
    culog.fit(X_train, y_train)

    score_test = 0.94
    score_train = 0.94

    assert culog.score(X_train, y_train) >= score_train
    assert culog.score(X_test, y_test) >= score_test


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logistic_regression_model_default(dtype):

    X_train, X_test, y_train, y_test = small_classification_dataset(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()
    culog.fit(X_train, y_train)
    sklog = skLog(multi_class="auto")

    sklog.fit(X_train, y_train)

    assert culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - 0.022


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("penalty", ["none", "l1", "l2"])
def test_logistic_regression_model_digits(
        dtype, order, sparse_input, fit_intercept, penalty):

    # smallest sklearn score with max_iter = 10000
    # put it as a constant here, because sklearn 0.23.1 needs a lot of iters
    # to converge and has a bug returning an unrelated error if not converged.
    acceptable_score = 0.95

    digits = load_digits()

    X_dense = digits.data.astype(dtype)
    X_dense.reshape(X_dense.shape, order=order)
    X = csr_matrix(X_dense) if sparse_input else X_dense

    y = digits.target.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    culog = cuLog(fit_intercept=fit_intercept, penalty=penalty)
    culog.fit(X_train, y_train)
    score = culog.score(X_test, y_test)

    assert score >= acceptable_score


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_logistic_regression_sparse_only(dtype, nlp_20news):

    # sklearn score with max_iter = 10000
    sklearn_score = 0.878
    acceptable_score = sklearn_score - 0.01

    X, y = nlp_20news

    X = csr_matrix(X.astype(dtype))
    y = y.get().astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    culog = cuLog()
    culog.fit(X_train, y_train)
    score = culog.score(X_test, y_test)

    assert score >= acceptable_score


@pytest.mark.parametrize("dtype, nrows, num_classes, fit_intercept", [
    (np.float32, 10, 2, True),
    (np.float64, 100, 10, False),
    (np.float64, 100, 2, True)
])
@pytest.mark.parametrize("column_info", [(20, 10)])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_logistic_regression_decision_function(
    dtype, nrows, column_info, num_classes, fit_intercept, sparse_input
):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype, nrows=nrows, ncols=ncols,
        n_info=n_info, num_classes=num_classes
    )
    X_train = csr_matrix(X_train) if sparse_input else X_train
    X_test = csr_matrix(X_test) if sparse_input else X_test

    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(fit_intercept=fit_intercept)
    sklog.coef_ = culog.coef_
    if fit_intercept:
        sklog.intercept_ = culog.intercept_
    else:
        skLog.intercept_ = 0
    sklog.classes_ = np.arange(num_classes)

    cu_dec_func = culog.decision_function(X_test)
    if num_classes > 2:
        cu_dec_func = cu_dec_func.T
    sk_dec_func = sklog.decision_function(X_test)

    assert array_equal(cu_dec_func, sk_dec_func)


@pytest.mark.parametrize("dtype, nrows, num_classes, fit_intercept", [
    (np.float32, 10, 2, True),
    (np.float64, 100, 10, False),
    (np.float64, 100, 2, True)
])
@pytest.mark.parametrize("column_info", [(20, 10)])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_logistic_regression_predict_proba(
    dtype, nrows, column_info, num_classes, fit_intercept, sparse_input
):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype, nrows=nrows, ncols=ncols,
        n_info=n_info, num_classes=num_classes
    )
    X_train = csr_matrix(X_train) if sparse_input else X_train
    X_test = csr_matrix(X_test) if sparse_input else X_test

    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    if num_classes > 2:
        sklog = skLog(
            fit_intercept=fit_intercept,
            solver="lbfgs",
            multi_class="multinomial"
        )
    else:
        sklog = skLog(fit_intercept=fit_intercept)
    sklog.coef_ = culog.coef_
    if fit_intercept:
        sklog.intercept_ = culog.intercept_
    else:
        skLog.intercept_ = 0
    sklog.classes_ = np.arange(num_classes)

    cu_proba = culog.predict_proba(X_test)
    sk_proba = sklog.predict_proba(X_test)

    cu_log_proba = culog.predict_log_proba(X_test)
    sk_log_proba = sklog.predict_log_proba(X_test)

    assert array_equal(cu_proba, sk_proba)
    assert array_equal(cu_log_proba, sk_log_proba)


@pytest.mark.parametrize("constructor", [np.array, cp.array, cudf.DataFrame])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logistic_regression_input_type_consistency(constructor, dtype):
    from cudf.core.frame import Frame

    X = constructor([[5, 10], [3, 1], [7, 8]]).astype(dtype)
    y = constructor([0, 1, 1]).astype(dtype)
    clf = cuLog().fit(X, y, convert_dtype=True)

    original_type = type(X)
    if constructor == cudf.DataFrame:
        original_type = Frame

    assert isinstance(clf.predict_proba(X), original_type)
    assert isinstance(clf.predict(X), original_type)


@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_linreg_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(n_samples=50, n_features=10,
                           n_informative=5, random_state=0)
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    clf = cuLinearRegression()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_ridge_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(n_samples=50, n_features=10,
                           n_informative=5, random_state=0)
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    clf = cuRidge()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_logistic_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_classification(n_samples=50, n_features=10, random_state=0)
    X = X.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    clf = cuLog()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.fixture(scope='session',
                params=['binary', 'multiclass-3', 'multiclass-7'])
def regression_dataset(request):
    regression_type = request.param

    out = {}
    for test_status in ['regular', 'stress_test']:
        if test_status == 'regular':
            n_samples, n_features = 100000, 5
        elif test_status == 'stress_test':
            n_samples, n_features = 1000000, 20

        data = (np.random.rand(n_samples, n_features) * 2) - 1

        if regression_type == 'binary':
            coef = (np.random.rand(n_features) * 2) - 1
            coef /= np.linalg.norm(coef)
            output = (data @ coef) > 0
        elif regression_type.startswith('multiclass'):
            n_classes = 3 if regression_type == 'multiclass-3' else 7
            coef = (np.random.rand(n_features, n_classes) * 2) - 1
            coef /= np.linalg.norm(coef, axis=0)
            output = (data @ coef).argmax(axis=1)
        output = output.astype(np.int32)

        out[test_status] = (regression_type, data, coef, output)
    return out


@pytest.mark.parametrize('option', ['sample_weight', 'class_weight',
                                    'balanced', 'no_weight'])
@pytest.mark.parametrize('test_status', ['regular',
                                         stress_param('stress_test')])
def test_logistic_regression_weighting(regression_dataset,
                                       option, test_status):
    regression_type, data, coef, output = regression_dataset[test_status]

    class_weight = None
    sample_weight = None
    if option == 'sample_weight':
        n_samples = data.shape[0]
        sample_weight = np.abs(np.random.rand(n_samples))
    elif option == 'class_weight':
        class_weight = np.random.rand(2)
        class_weight = {0: class_weight[0], 1: class_weight[1]}
    elif option == 'balanced':
        class_weight = 'balanced'

    culog = cuLog(fit_intercept=False, class_weight=class_weight)
    culog.fit(data, output, sample_weight=sample_weight)

    sklog = skLog(fit_intercept=False, class_weight=class_weight)
    sklog.fit(data, output, sample_weight=sample_weight)

    skcoef = np.squeeze(sklog.coef_)
    cucoef = np.squeeze(culog.coef_)
    if regression_type == 'binary':
        skcoef /= np.linalg.norm(skcoef)
        cucoef /= np.linalg.norm(cucoef)
        unit_tol = 0.04
        total_tol = 0.08
    elif regression_type.startswith('multiclass'):
        skcoef /= np.linalg.norm(skcoef, axis=1)[:, None]
        cucoef /= np.linalg.norm(cucoef, axis=1)[:, None]
        unit_tol = 0.2
        total_tol = 0.3

    equality = array_equal(skcoef, cucoef, unit_tol=unit_tol,
                           total_tol=total_tol)
    if not equality:
        print('\ncoef.shape: ', coef.shape)
        print('coef:\n', coef)
        print('cucoef.shape: ', cucoef.shape)
        print('cucoef:\n', cucoef)
    assert equality

    cuOut = culog.predict(data)
    skOut = sklog.predict(data)
    assert array_equal(skOut, cuOut, unit_tol=unit_tol,
                       total_tol=total_tol)


@pytest.mark.parametrize('algo', [cuLog, cuRidge])
# ignoring warning about change of solver
@pytest.mark.filterwarnings("ignore::UserWarning:cuml[.*]")
def test_linear_models_set_params(algo):
    x = np.linspace(0, 1, 50)
    y = 2 * x

    model = algo()
    model.fit(x, y)
    coef_before = model.coef_

    if algo == cuLog:
        params = {'penalty': "none", 'C': 1, 'max_iter': 30}
        model = algo(penalty='none', C=1, max_iter=30)
    else:
        model = algo(solver='svd', alpha=0.1)
        params = {'solver': "svd", 'alpha': 0.1}
    model.fit(x, y)
    coef_after = model.coef_

    model = algo()
    model.set_params(**params)
    model.fit(x, y)
    coef_test = model.coef_

    assert not array_equal(coef_before, coef_after)
    assert array_equal(coef_after, coef_test)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
@pytest.mark.parametrize(
    "nrows", [unit_param(1000), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500])
    ],
)
def test_elasticnet_solvers_eq(datatype, alpha, l1_ratio, nrows, column_info):

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    kwargs = {'alpha': alpha, 'l1_ratio': l1_ratio}
    cd = cuElasticNet(solver='cd', **kwargs)
    cd.fit(X_train, y_train)
    cd_res = cd.predict(X_test)

    qn = cuElasticNet(solver='qn', **kwargs)
    qn.fit(X_train, y_train)
    # the results of the two models should be close (even if both are bad)
    assert qn.score(X_test, cd_res) > 0.95
    # coefficients of the two models should be close
    assert np.corrcoef(cd.coef_, qn.coef_)[0, 1] > 0.98
