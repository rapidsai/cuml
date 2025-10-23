# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
import sklearn
from hypothesis import assume, example, given, note
from hypothesis import strategies as st
from hypothesis import target
from packaging.version import Version
from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import ElasticNet as skElasticNet
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import LogisticRegression as skLog
from sklearn.linear_model import Ridge as skRidge
from sklearn.model_selection import train_test_split

import cuml
from cuml import ElasticNet as cuElasticNet
from cuml import LinearRegression as cuLinearRegression
from cuml import LogisticRegression as cuLog
from cuml import Ridge as cuRidge
from cuml.testing.datasets import (
    is_cuml_compatible_dataset,
    is_sklearn_compatible_dataset,
    make_classification,
    make_classification_dataset,
    make_regression,
    make_regression_dataset,
    regression_datasets,
    small_classification_dataset,
    small_regression_dataset,
    split_datasets,
    standard_classification_datasets,
    standard_regression_datasets,
)
from cuml.testing.strategies import dataset_dtypes
from cuml.testing.utils import array_difference, array_equal


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd"]),
    nrows=st.integers(min_value=1000, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
    ntargets=st.integers(min_value=1, max_value=2),
)
@example(
    datatype=np.float32,
    algorithm="eig",
    nrows=1000,
    column_info=[20, 10],
    ntargets=1,
)
@example(
    datatype=np.float64,
    algorithm="svd",
    nrows=5000,
    column_info=[100, 50],
    ntargets=2,
)
def test_linear_regression_model(
    datatype, algorithm, nrows, column_info, ntargets
):
    if algorithm == "svd" and nrows > 46340:
        pytest.skip(
            "svd solver is not supported for the data that has more"
            "than 46340 rows or columns if you are using CUDA version"
            "10.x"
        )
    if 1 < ntargets and algorithm != "svd":
        pytest.skip("The multi-target fit only supports using the svd solver.")

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, n_targets=ntargets
    )

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=True, algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    if nrows < 500000:
        # sklearn linear regression model initialization, fit and predict
        skols = skLinearRegression(fit_intercept=True)
        skols.fit(X_train, y_train)

        skols_predict = skols.predict(X_test)

        assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)


@given(
    ntargets=st.integers(min_value=1, max_value=2),
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd", "qr", "svd-qr"]),
    fit_intercept=st.booleans(),
    distribution=st.sampled_from(["lognormal", "exponential", "uniform"]),
)
@example(
    ntargets=1,
    datatype=np.float32,
    algorithm="eig",
    fit_intercept=True,
    distribution="uniform",
)
@example(
    ntargets=2,
    datatype=np.float64,
    algorithm="svd",
    fit_intercept=False,
    distribution="lognormal",
)
def test_weighted_linear_regression(
    ntargets, datatype, algorithm, fit_intercept, distribution
):
    nrows, ncols, n_info = 1000, 20, 10
    max_weight = 10
    noise = 20

    if 1 < ntargets and algorithm != "svd":
        pytest.skip("The multi-target fit only supports using the svd solver.")

    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, noise=noise, n_targets=ntargets
    )

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=max_weight, size=len(X_train))
        wt_test = np.random.randint(1, high=max_weight, size=len(X_test))
    elif distribution == "exponential":
        wt = np.random.exponential(size=len(X_train))
        wt_test = np.random.exponential(size=len(X_test))
    else:
        wt = np.random.lognormal(size=len(X_train))
        wt_test = np.random.lognormal(size=len(X_test))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(
        fit_intercept=fit_intercept, algorithm=algorithm
    )

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train, sample_weight=wt)
    cuols_predict = cuols.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skols = skLinearRegression(fit_intercept=fit_intercept)
    skols.fit(X_train, y_train, sample_weight=wt)

    skols_predict = skols.predict(X_test)

    assert array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)

    # Compare weighted scores
    sk_score = skols.score(X_test, y_test, sample_weight=wt_test)
    cu_score = cuols.score(X_test, y_test, sample_weight=wt_test)
    np.testing.assert_almost_equal(cu_score, sk_score)


def test_linear_regression_single_column():
    """Test that linear regression can be run on single column with more than
    46340 rows"""
    model = cuml.LinearRegression()
    model.fit(cp.random.rand(46341)[:, None], cp.random.rand(46341))


# The assumptions required to have this test pass are relatively strong.
# It should be possible to relax assumptions once #4963 is resolved.
# See also: test_linear_regression_model_default_generalized
@given(
    split_datasets(
        standard_regression_datasets(
            n_samples=st.just(1000),
            n_targets=st.integers(1, 10),
        ),
        test_sizes=st.just(0.2),
    )
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_linear_regression_model_default(dataset):

    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(is_sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(is_cuml_compatible_dataset(X_train, X_test, y_train))

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
@given(split_datasets(regression_datasets()))
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_linear_regression_model_default_generalized(dataset):

    X_train, X_test, y_train, _ = dataset

    # Filter datasets based on required assumptions
    assume(is_sklearn_compatible_dataset(X_train, X_test, y_train))
    assume(is_cuml_compatible_dataset(X_train, X_test, y_train))

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


@given(
    split_datasets(
        standard_regression_datasets(),
    ),
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
def test_ridge_regression_model_default(dataset):

    assume(is_sklearn_compatible_dataset(*dataset))
    assume(is_cuml_compatible_dataset(*dataset))
    X_train, X_test, y_train, _ = dataset

    curidge = cuRidge()

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    # sklearn ridge regression model initialization, fit and predict
    skridge = skRidge()
    skridge.fit(X_train, y_train)
    skridge_predict = skridge.predict(X_test)

    equal = array_equal(
        skridge_predict,
        curidge_predict,
        unit_tol=1e-1,
        total_tol=1e-3,
        with_sign=True,
    )
    note(equal)
    target(float(np.abs(equal.compute_difference())))
    assert equal


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd"]),
    nrows=st.integers(min_value=500, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
)
@example(datatype=np.float32, algorithm="eig", nrows=500, column_info=[20, 10])
@example(
    datatype=np.float64, algorithm="svd", nrows=5000, column_info=[100, 50]
)
def test_ridge_regression_model(datatype, algorithm, nrows, column_info):

    if algorithm == "svd" and nrows > 46340:
        pytest.skip(
            "svd solver is not supported for the data that has more"
            "than 46340 rows or columns if you are using CUDA version"
            "10.x"
        )

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    # Initialization of cuML's ridge regression model
    curidge = cuRidge(fit_intercept=False, solver=algorithm)

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    if nrows < 500000:
        # sklearn ridge regression model initialization, fit and predict
        skridge = skRidge(fit_intercept=False)
        skridge.fit(X_train, y_train)

        skridge_predict = skridge.predict(X_test)

        assert array_equal(
            skridge_predict, curidge_predict, 1e-1, with_sign=True
        )


def test_ridge_and_least_squares_equal_when_alpha_is_0():
    X, y = make_regression(n_samples=5, n_features=4, random_state=0)

    ridge = cuRidge(alpha=0.0, fit_intercept=False)
    ols = cuLinearRegression(fit_intercept=False)

    ridge.fit(X, y)
    ols.fit(X, y)
    assert array_equal(ridge.coef_, ols.coef_)


@given(
    datatype=dataset_dtypes(),
    algorithm=st.sampled_from(["eig", "svd"]),
    fit_intercept=st.booleans(),
    distribution=st.sampled_from(["lognormal", "exponential", "uniform"]),
)
@example(
    datatype=np.float32,
    algorithm="eig",
    fit_intercept=True,
    distribution="uniform",
)
@example(
    datatype=np.float64,
    algorithm="svd",
    fit_intercept=False,
    distribution="lognormal",
)
def test_weighted_ridge(datatype, algorithm, fit_intercept, distribution):
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
    curidge = cuRidge(fit_intercept=fit_intercept, solver=algorithm)

    # fit and predict cuml linear regression model
    curidge.fit(X_train, y_train, sample_weight=wt)
    curidge_predict = curidge.predict(X_test)

    # sklearn linear regression model initialization, fit and predict
    skridge = skRidge(fit_intercept=fit_intercept)
    skridge.fit(X_train, y_train, sample_weight=wt)

    skridge_predict = skridge.predict(X_test)

    assert array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)


@given(
    num_classes=st.integers(min_value=2, max_value=10),
    dtype=dataset_dtypes(),
    penalty=st.sampled_from([None, "l1", "l2", "elasticnet"]),
    l1_ratio=st.floats(min_value=0.0, max_value=1.0),
    fit_intercept=st.booleans(),
    nrows=st.integers(min_value=1000, max_value=5000),
    C=st.floats(min_value=0.5, max_value=2.0),
    tol=st.floats(min_value=1e-8, max_value=1e-3),
)
@example(
    num_classes=2,
    dtype=np.float32,
    penalty=None,
    l1_ratio=1.0,
    fit_intercept=True,
    nrows=1000,
    C=1.0,
    tol=1e-3,
)
@example(
    num_classes=10,
    dtype=np.float64,
    penalty="l2",
    l1_ratio=1.0,
    fit_intercept=True,
    nrows=5000,
    C=1.0,
    tol=1e-8,
)
@example(
    num_classes=10,
    dtype=np.float32,
    penalty="elasticnet",
    l1_ratio=0.0,
    fit_intercept=True,
    nrows=1000,
    C=1.0,
    tol=1e-3,
)
# ignoring UserWarnings in sklearn about setting unused parameters
# like l1 for none penalty
@pytest.mark.filterwarnings("ignore::UserWarning:sklearn[.*]")
def test_logistic_regression(
    num_classes,
    dtype,
    penalty,
    l1_ratio,
    fit_intercept,
    nrows,
    C,
    tol,
):
    ncols, n_info = 20, 10
    # Checking sklearn >= 0.21 for testing elasticnet
    sk_check = Version(str(sklearn.__version__)) >= Version("0.21.0")
    if not sk_check and penalty == "elasticnet":
        pytest.skip(
            "Need sklearn > 0.21 for testing logistic with" "elastic net."
        )

    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype=dtype,
        nrows=nrows,
        ncols=ncols,
        n_info=n_info,
        num_classes=num_classes,
    )
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    culog = cuLog(
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        fit_intercept=fit_intercept,
        tol=tol,
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
    cu_preds = culog.predict(X_test)

    tol_score = 1e-1
    assert (
        culog.score(X_train, y_train)
        >= sklog.score(X_train, y_train) - tol_score
    )
    assert (
        culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - tol_score
    )
    assert len(np.unique(cu_preds)) == len(np.unique(y_test))

    if fit_intercept is False:
        assert np.array_equal(culog.intercept_, sklog.intercept_)


@given(
    dtype=dataset_dtypes(),
    l1_ratio=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    penalty=st.sampled_from([None, "l1", "l2", "elasticnet"]),
)
@example(dtype=np.float32, l1_ratio=None, penalty=None)
@example(dtype=np.float64, l1_ratio=None, penalty="l1")
@example(dtype=np.float64, l1_ratio=0.5, penalty="elasticnet")
@example(dtype=np.float64, l1_ratio=0.5, penalty="l2")
def test_logistic_regression_unscaled(dtype, penalty, l1_ratio):
    if penalty == "elasticnet":
        assume(l1_ratio is not None)

    # Test logistic regression on the breast cancer dataset. We do not scale
    # the dataset which could lead to numerical problems (fixed in PR #2543).
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    params = {
        "penalty": penalty,
        "C": 1,
        "tol": 1e-4,
        "fit_intercept": True,
        "max_iter": 5000,
        "l1_ratio": l1_ratio,
    }
    culog = cuLog(**params)
    culog.fit(X_train, y_train)

    score_train = culog.score(X_train, y_train)
    score_test = culog.score(X_test, y_test)

    target(1 / score_train, label="inverse train score")
    target(1 / score_test, label="inverse test score")

    # TODO: Use a more rigorous approach to determine expected minimal scores
    # here. The values here are selected empirically and passed during test
    # development.
    assert score_train >= 0.94
    assert score_test >= 0.94


@given(dtype=dataset_dtypes())
@example(dtype=np.float32)
@example(dtype=np.float64)
def test_logistic_regression_model_default(dtype):

    X_train, X_test, y_train, y_test = small_classification_dataset(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)
    culog = cuLog()
    culog.fit(X_train, y_train)
    sklog = skLog(multi_class="auto")

    sklog.fit(X_train, y_train)

    assert culog.score(X_test, y_test) >= sklog.score(X_test, y_test) - 0.022


@given(
    dtype=dataset_dtypes(),
    order=st.sampled_from(["C", "F"]),
    sparse_input=st.booleans(),
    fit_intercept=st.booleans(),
    penalty=st.sampled_from([None, "l1", "l2"]),
)
@example(
    dtype=np.float32,
    order="C",
    sparse_input=False,
    fit_intercept=True,
    penalty=None,
)
@example(
    dtype=np.float64,
    order="F",
    sparse_input=True,
    fit_intercept=False,
    penalty="l1",
)
@example(
    dtype=np.float32,
    order="C",
    sparse_input=True,
    fit_intercept=False,
    penalty="l2",
)
def test_logistic_regression_model_digits(
    dtype, order, sparse_input, fit_intercept, penalty
):

    # smallest sklearn score with max_iter = 10000
    # put it as a constant here, because sklearn 0.23.1 needs a lot of iters
    # to converge and has a bug returning an unrelated error if not converged.
    acceptable_score = 0.9

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


@given(dtype=dataset_dtypes())
@example(dtype=np.float32)
@example(dtype=np.float64)
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


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@example(
    dataset=small_classification_dataset(np.float32),
    fit_intercept=True,
    sparse_input=False,
)
@example(
    dataset=small_classification_dataset(np.float64),
    fit_intercept=False,
    sparse_input=True,
)
def test_logistic_regression_decision_function(
    dataset, fit_intercept, sparse_input
):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(fit_intercept=fit_intercept)
    sklog.coef_ = culog.coef_
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0
    sklog.classes_ = np.arange(num_classes)

    cu_dec_func = culog.decision_function(X_test)
    sk_dec_func = sklog.decision_function(X_test)

    assert array_equal(cu_dec_func, sk_dec_func)


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@example(
    dataset=small_classification_dataset(np.float32),
    fit_intercept=True,
    sparse_input=False,
)
@example(
    dataset=small_classification_dataset(np.float64),
    fit_intercept=False,
    sparse_input=True,
)
def test_logistic_regression_predict_proba(
    dataset, fit_intercept, sparse_input
):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))
    num_classes = len(np.unique(y_train))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(
        fit_intercept=fit_intercept,
        **(
            {"solver": "lbfgs", "multi_class": "multinomial"}
            if num_classes > 2
            else {}
        ),
    )
    sklog.coef_ = culog.coef_
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0
    sklog.classes_ = np.arange(num_classes)

    cu_proba = culog.predict_proba(X_test)
    sk_proba = sklog.predict_proba(X_test)

    cu_log_proba = culog.predict_log_proba(X_test)
    sk_log_proba = sklog.predict_log_proba(X_test)

    assert array_equal(cu_proba, sk_proba)

    # if the probabilities pass test, then the margin of the logarithm
    # of the probabilities can be relaxed to avoid false positives.
    assert array_equal(
        cu_log_proba, sk_log_proba, unit_tol=1e-2, total_tol=1e-3
    )


@pytest.mark.parametrize("constructor", [np.array, cp.array, cudf.DataFrame])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_logistic_regression_input_type_consistency(constructor, dtype):
    X = constructor([[5, 10], [3, 1], [7, 8]]).astype(dtype)
    y = constructor([0, 1, 1]).astype(dtype)
    clf = cuLog().fit(X, y, convert_dtype=True)

    assert isinstance(clf.predict_proba(X), type(X))
    expected_type = cudf.Series if constructor == cudf.DataFrame else type(X)
    assert isinstance(clf.predict(X), expected_type)


@pytest.mark.parametrize(
    "y_kind", ["object", "fixed-string", "int32", "float32", "float16"]
)
@pytest.mark.parametrize("output_type", ["numpy", "cupy", "cudf", "pandas"])
def test_logistic_regression_complex_classes(y_kind, output_type):
    """Test that LogisticRegression handles non-numeric or non-monotonically
    increasing classes properly in both `fit` and `predict`"""
    if output_type == "cupy" and y_kind in ("object", "fixed-string"):
        pytest.skip("cupy doesn't support strings!")
    elif output_type in ("cudf", "pandas") and y_kind == "float16":
        pytest.skip("float16 dtype not supported")

    X, y_inds = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=0,
    )
    if y_kind == "object":
        classes = np.array(["apple", "banana", "carrot"], dtype="object")
        df_dtype = "object"
    elif y_kind == "fixed-string":
        classes = np.array(["apple", "banana", "carrot"], dtype="U")
        df_dtype = "object"
    else:
        classes = np.array([10, 20, 30], dtype=y_kind)
        df_dtype = classes.dtype

    y = classes.take(y_inds)

    cu_model = cuLog(output_type=output_type)
    sk_model = skLog()

    cu_model.fit(X, y)
    sk_model.fit(X, y)

    np.testing.assert_array_equal(
        cu_model.classes_, sk_model.classes_, strict=True
    )

    res = cu_model.predict(X)
    sol = sk_model.predict(X)
    if output_type == "numpy":
        assert res.dtype == sol.dtype
        assert isinstance(res, np.ndarray)
    elif output_type == "cupy":
        assert res.dtype == sol.dtype
        assert isinstance(res, cp.ndarray)
    elif output_type == "pandas":
        assert res.dtype == df_dtype
        assert isinstance(res, pd.Series)
    elif output_type == "cudf":
        assert res.dtype == df_dtype
        assert isinstance(res, cudf.Series)


@pytest.mark.parametrize("y_kind", ["pandas", "cudf"])
def test_logistic_regression_categorical_y(y_kind):
    X, y_inds = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_classes=3,
        random_state=0,
    )
    categories = np.array(["apple", "banana", "carrot"], dtype="object")
    y = pd.Series(pd.Categorical.from_codes(y_inds, categories))
    if y_kind == "cudf":
        y = cudf.Series(y)

    model = cuLog(output_type="numpy")
    model.fit(X, y)
    np.testing.assert_array_equal(model.classes_, categories, strict=True)
    res = model.predict(X)
    assert isinstance(res, np.ndarray)
    assert res.dtype == "object"


@given(train_dtype=dataset_dtypes(), test_dtype=dataset_dtypes())
@example(train_dtype=np.float32, test_dtype=np.float32)
@example(train_dtype=np.float32, test_dtype=np.float64)
@example(train_dtype=np.float64, test_dtype=np.float32)
@example(train_dtype=np.float64, test_dtype=np.float64)
def test_linreg_predict_convert_dtype(train_dtype, test_dtype):
    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=0
    )
    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    clf = cuLinearRegression()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@given(
    dataset=split_datasets(
        standard_regression_datasets(),
    ),
    test_dtype=dataset_dtypes(),
)
@example(dataset=small_regression_dataset(np.float32), test_dtype=np.float32)
@example(dataset=small_regression_dataset(np.float32), test_dtype=np.float64)
@example(dataset=small_regression_dataset(np.float64), test_dtype=np.float32)
@example(dataset=small_regression_dataset(np.float64), test_dtype=np.float64)
def test_ridge_predict_convert_dtype(dataset, test_dtype):
    assume(is_cuml_compatible_dataset(*dataset))
    X_train, X_test, y_train, _ = dataset

    clf = cuRidge()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@given(
    dataset=split_datasets(
        standard_classification_datasets(),
    ),
    test_dtype=dataset_dtypes(),
)
@example(
    dataset=small_classification_dataset(np.float32), test_dtype=np.float32
)
@example(
    dataset=small_classification_dataset(np.float32), test_dtype=np.float64
)
@example(
    dataset=small_classification_dataset(np.float64), test_dtype=np.float32
)
@example(
    dataset=small_classification_dataset(np.float64), test_dtype=np.float64
)
def test_logistic_predict_convert_dtype(dataset, test_dtype):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))

    clf = cuLog()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@given(
    dataset=standard_classification_datasets(),
    use_sample_weight=st.booleans(),
    class_weight_option=st.sampled_from([None, "balanced", "dict"]),
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=2,
            n_redundant=0,
            random_state=0,
        ),
    ),
    use_sample_weight=True,
    class_weight_option=None,
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=7,
            n_redundant=0,
            random_state=1,
        ),
    ),
    use_sample_weight=False,
    class_weight_option="balanced",
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=3,
            n_redundant=0,
            random_state=2,
        ),
    ),
    use_sample_weight=True,
    class_weight_option="dict",
)
@example(
    dataset=(
        *make_classification(
            n_samples=100000,
            n_features=5,
            n_informative=4,
            n_classes=2,
            n_redundant=0,
            random_state=3,
        ),
    ),
    use_sample_weight=False,
    class_weight_option=None,
)
def test_logistic_regression_weighting(
    dataset, use_sample_weight, class_weight_option
):
    X, y = dataset

    num_classes = len(np.unique(y))

    # Set up sample_weight
    sample_weight = None
    if use_sample_weight:
        n_samples = X.shape[0]
        sample_weight = np.abs(np.random.rand(n_samples))

    # Set up class_weight
    class_weight = None
    match class_weight_option:
        case "dict":
            weights = np.random.rand(num_classes)
            class_weight = {i: weights[i] for i in range(num_classes)}
        case "balanced":
            class_weight = "balanced"
        case None:
            pass
        case _:
            raise ValueError(
                f"Unknown class_weight_option: {class_weight_option}"
            )

    # Use higher max_iter for better convergence with complex weighting
    max_iter = (
        1000
        if (use_sample_weight and class_weight_option is not None)
        else 500
    )

    culog = cuLog(
        fit_intercept=False, class_weight=class_weight, max_iter=max_iter
    )
    culog.fit(X, y, sample_weight=sample_weight)

    sklog = skLog(
        fit_intercept=False, class_weight=class_weight, max_iter=max_iter
    )
    sklog.fit(X, y, sample_weight=sample_weight)

    skcoef = np.squeeze(sklog.coef_)
    cucoef = np.squeeze(culog.coef_)

    # Normalize coefficients
    if num_classes == 2:
        skcoef /= np.linalg.norm(skcoef)
        cucoef /= np.linalg.norm(cucoef)
    else:
        skcoef /= np.linalg.norm(skcoef, axis=1, keepdims=True)
        cucoef /= np.linalg.norm(cucoef, axis=1, keepdims=True)

    # Set tolerances based on empirical analysis
    has_weights = use_sample_weight or class_weight_option is not None
    harder_problem = has_weights or num_classes > 3
    unit_tol = 0.25 if harder_problem else 0.04
    total_tol = 0.40 if harder_problem else 0.08
    assert array_equal(skcoef, cucoef, unit_tol=unit_tol, total_tol=total_tol)

    cuOut = culog.predict(X)
    skOut = sklog.predict(X)
    assert array_equal(skOut, cuOut, unit_tol=unit_tol, total_tol=total_tol)


@pytest.mark.parametrize("algo", [cuLog, cuRidge])
# ignoring warning about change of solver
@pytest.mark.filterwarnings("ignore::UserWarning:cuml[.*]")
def test_linear_models_set_params(algo):
    x = np.linspace(0, 1, 50)[:, None]
    y = 2 * x

    model = algo()
    model.fit(x, y)
    coef_before = model.coef_

    if algo == cuLog:
        params = {"penalty": None, "C": 1, "max_iter": 30}
        model = algo(penalty=None, C=1, max_iter=30)
    else:
        model = algo(solver="svd", alpha=0.1)
        params = {"solver": "svd", "alpha": 0.1}
    model.fit(x, y)
    coef_after = model.coef_

    model = algo()
    model.set_params(**params)
    model.fit(x, y)
    coef_test = model.coef_

    assert not array_equal(coef_before, coef_after)
    assert array_equal(coef_after, coef_test)


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
def test_elasticnet_solvers_eq(datatype, alpha, l1_ratio, nrows, column_info):

    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info
    )

    kwargs = {"alpha": alpha, "l1_ratio": l1_ratio}
    cd = cuElasticNet(solver="cd", **kwargs)
    cd.fit(X_train, y_train)
    cd_res = cd.predict(X_test)

    qn = cuElasticNet(solver="qn", **kwargs)
    qn.fit(X_train, y_train)
    # the results of the two models should be close (even if both are bad)
    assert qn.score(X_test, cd_res) > 0.90
    # coefficients of the two models should be close
    assert np.corrcoef(cd.coef_, qn.coef_)[0, 1] > 0.98


@pytest.mark.filterwarnings("ignore:Changing solver.*:UserWarning")
@given(
    algo=st.sampled_from(["eig", "qr", "svd", "svd-qr"]),
    n_targets=st.integers(min_value=1, max_value=2),
    fit_intercept=st.booleans(),
    weighted=st.booleans(),
)
@example(algo="eig", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="qr", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd-qr", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd", n_targets=1, fit_intercept=True, weighted=False)
@example(algo="svd", n_targets=2, fit_intercept=False, weighted=True)
def test_linear_regression_input_mutation(
    algo, n_targets, fit_intercept, weighted
):
    """Check that `LinearRegression.fit`:
    - Never mutates y and sample_weight
    - Only sometimes mutates X
    """
    # Only algo="svd" supports n_targets > 1. While we do fallback to svd
    # automatically (with a warning), there's no need to have hypothesis
    # explore those cases.
    assume(n_targets == 1 or algo == "svd")

    X, y = make_regression(n_targets=n_targets, random_state=42)
    if weighted:
        sample_weight = (
            cp.random.default_rng(42)
            .uniform(0.5, 1, y.shape[0])
            .astype("float32")
        )
        sample_weight_orig = sample_weight.copy()
    else:
        sample_weight = None

    # The solvers expected fortran-ordered inputs, and will always copy C
    # ordered inputs. Mutation can only happen for F-ordered inputs.
    X = cp.asarray(X, order="F", dtype="float32")
    y = cp.asarray(y, order="F", dtype="float32")
    X_orig = X.copy()
    y_orig = y.copy()

    params = {"algorithm": algo, "fit_intercept": fit_intercept}

    # Default never mutates inputs
    cuml.LinearRegression(**params).fit(X, y, sample_weight=sample_weight)
    cp.testing.assert_allclose(X, X_orig)
    cp.testing.assert_allclose(y, y_orig)
    if weighted:
        cp.testing.assert_allclose(sample_weight, sample_weight_orig)

    cuml.LinearRegression(copy_X=False, **params).fit(
        X, y, sample_weight=sample_weight
    )
    # y and sample_weight are never mutated
    cp.testing.assert_allclose(y, y_orig)
    if weighted:
        cp.testing.assert_allclose(sample_weight, sample_weight_orig)
    # The interface doesn't actually care if X is mutated if copy_X=False,
    # but if our solvers stop mutating (and we can avoid a copy) it'd be good
    # to notice. Asserting the current behavior here for now.
    if n_targets == 1 and algo in ["eig", "qr", "svd", "svd-qr"]:
        # `eig` sometimes mutates and sometimes doesn't, the others always do
        if algo != "eig":
            assert not cp.array_equal(X, X_orig)
    elif n_targets > 1 and not fit_intercept and weighted:
        # The fallback solver also mutates in this case
        assert not cp.array_equal(X, X_orig)
    else:
        # All other options don't mutate
        cp.testing.assert_array_equal(X, X_orig)


@given(
    ntargets=st.integers(min_value=1, max_value=2),
    datatype=dataset_dtypes(),
    solver=st.sampled_from(["cd", "qn"]),
    nrows=st.integers(min_value=1000, max_value=5000),
    column_info=st.sampled_from([[20, 10], [100, 50]]),
)
@example(
    ntargets=1,
    datatype=np.float32,
    solver="cd",
    nrows=1000,
    column_info=[20, 10],
)
@example(
    ntargets=2,
    datatype=np.float64,
    solver="qn",
    nrows=5000,
    column_info=[100, 50],
)
def test_elasticnet_model(datatype, solver, nrows, column_info, ntargets):
    ncols, n_info = column_info
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols, n_info, n_targets=ntargets
    )

    # Initialization of cuML's elastic net model
    cuelastic = cuElasticNet(alpha=0.1, l1_ratio=0.5, solver=solver)

    if ntargets > 1:
        with pytest.raises(
            ValueError,
            match="The .* solver does not support multi-target regression.",
        ):
            cuelastic.fit(X_train, y_train)
        return

    # fit and predict cuml elastic net model
    cuelastic.fit(X_train, y_train)
    cuelastic_predict = cuelastic.predict(X_test)

    if nrows < 500000:
        # sklearn elastic net model initialization, fit and predict
        skelastic = skElasticNet(alpha=0.1, l1_ratio=0.5)
        skelastic.fit(X_train, y_train)

        skelastic_predict = skelastic.predict(X_test)

        assert array_equal(
            skelastic_predict,
            cuelastic_predict,
            3e-0,
            total_tol=1e-0,
            with_sign=True,
        )
