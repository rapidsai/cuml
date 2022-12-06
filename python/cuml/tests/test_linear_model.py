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
from functools import partial
import cupy as cp
import numpy as np
import pytest
from hypothesis import (
    assume,
    event,
    example,
    given,
    note,
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
from cuml.common.input_utils import _typecast_will_lose_information
from cuml.testing.strategies import (
    combined_datasets_strategy,
    standard_classification_datasets,
    standard_regression_datasets,
    regression_datasets,
    split_datasets,
    standard_datasets,
)
from cuml.testing.utils import (
    array_difference,
    array_equal,
    small_regression_dataset,
    unit_param,
    quality_param,
    stress_param,
)
import rmm

from scipy.sparse import csr_matrix

import sklearn
from sklearn.datasets import make_regression
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import LogisticRegression as skLog
from sklearn.model_selection import train_test_split


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
        and not any(_typecast_will_lose_information(x, np.float32)
                    for x in (X_train, X_test, y_train) if x is not None)
    )


@pytest.mark.xfail(reason="https://github.com/rapidsai/cuml/issues/4963")
@pytest.mark.parametrize(
    "n_samples", [unit_param(1000), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_features", [unit_param(20), quality_param(100), stress_param(1000)]
)
@given(data=st.data(), algorithm=st.sampled_from(("eig", "svd")))
@settings(deadline=5000)
def test_linear_regression_model(data, algorithm, n_samples, n_features):

    # svd solver is not supported for the data that has more than 46340 rows or
    # columns if you are using CUDA version 10.x.
    assume(not (algorithm == "svd" and n_samples > 46340))

    X_train, X_test, y_train, _ = data.draw(
        split_datasets(
            regression_datasets(
                n_samples=st.integers(min_value=0, max_value=n_samples),
                n_features=st.integers(min_value=0, max_value=n_features),
                n_targets=st.sampled_from((1, 2)),
                n_informatives=st.integers(min_value=0, max_value=n_features),
                dtypes=floating_dtypes(sizes=(32, 64))
            )
        )
    )

    # Filter datasets based on required assumptions
    assume(cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's linear regression model
    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    # fit and predict cuml linear regression model
    cuols.fit(X_train, y_train)
    cuols_predict = cuols.predict(X_test)

    if n_samples < 500000:
        assume(sklearn_compatible_dataset(X_train, X_test, y_train))

        # sklearn linear regression model initialization, fit and predict
        skols = skLinearRegression(fit_intercept=True, normalize=False)
        skols.fit(X_train, y_train)

        skols_predict = skols.predict(X_test)

        equal = array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)
        note(equal)
        target(float(np.abs(equal.compute_difference())))
        assert equal


@pytest.mark.xfail(reason="https://github.com/rapidsai/cuml/issues/4963")
@given(
    datasets=split_datasets(
        combined_datasets_strategy(
            standard_datasets,
            partial(standard_regression_datasets, noise=st.just(20)),
        )(
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_targets=st.sampled_from((1, 2)),
        )
    ),
    algorithm=st.sampled_from(("eig", "svd", "qr", "svd-qr")),
    fit_intercept=st.booleans(),
    normalize=st.booleans(),
    distribution=st.sampled_from(("lognormal", "exponential", "uniform")),
)
@settings(deadline=5000)
def test_weighted_linear_regression(datasets, algorithm, fit_intercept,
                                    normalize, distribution):

    assume(cuml_compatible_dataset(* datasets))
    assume(sklearn_compatible_dataset(* datasets))
    X_train, X_test, y_train, _ = datasets

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=10, size=len(X_train))
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

    equal = array_equal(skols_predict, cuols_predict, 1e-1, with_sign=True)
    note(equal)
    target(float(np.abs(equal.compute_difference())))
    assert equal


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
            n_targets=st.integers(1, 10),
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
@pytest.mark.xfail(reason="https://github.com/rapidsai/cuml/issues/4963")
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


# TODO: create issue
@pytest.mark.xfail(reason="Difference is too large for some inputs.")
@given(
    split_datasets(regression_datasets(dtypes=floating_dtypes(sizes=(32, 64))))
)
@example(small_regression_dataset(np.float32))
@example(small_regression_dataset(np.float64))
@settings(deadline=5000)
def test_ridge_regression_model_default(dataset):

    assume(sklearn_compatible_dataset(* dataset))
    assume(cuml_compatible_dataset(* dataset))
    X_train, X_test, y_train, _ = dataset

    curidge = cuRidge()

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    # sklearn ridge regression model initialization, fit and predict
    skridge = skRidge()
    skridge.fit(X_train, y_train)
    skridge_predict = skridge.predict(X_test)

    equal = array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)
    note(equal)
    target(float(np.abs(equal.compute_difference())))
    assert equal


@pytest.mark.xfail(reason="RuntimeError: cuSOLVER error")  # TODO: create issue
@pytest.mark.parametrize(
    "n_samples", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_features", [unit_param(20), quality_param(100), stress_param(1000)]
)
@given(data=st.data(), algorithm=st.sampled_from(("eig", "svd")))
def test_ridge_regression_model(data, algorithm, n_samples, n_features):

    # svd solver is not supported for the data that has more than 46340 rows or
    # columns if you are using CUDA version 10.x.
    assume(not (algorithm == "svd" and n_samples > 46340))

    X_train, X_test, y_train, _ = data.draw(
        split_datasets(
            regression_datasets(
                n_samples=st.integers(min_value=0, max_value=n_samples),
                n_features=st.integers(min_value=0, max_value=n_features),
                dtypes=floating_dtypes(sizes=(32, 64))
            )
        )
    )

    # Filter datasets based on required assumptions
    assume(cuml_compatible_dataset(X_train, X_test, y_train))

    # Initialization of cuML's ridge regression model
    curidge = cuRidge(fit_intercept=False, normalize=False, solver=algorithm)

    # fit and predict cuml ridge regression model
    curidge.fit(X_train, y_train)
    curidge_predict = curidge.predict(X_test)

    if n_samples < 500000:
        assume(sklearn_compatible_dataset(X_train, X_test, y_train))

        # sklearn ridge regression model initialization, fit and predict
        skridge = skRidge(fit_intercept=False, normalize=False)
        skridge.fit(X_train, y_train)

        skridge_predict = skridge.predict(X_test)

        equal = array_equal(skridge_predict, curidge_predict, 1e-1)
        note(equal)
        target(float(np.abs(equal.compute_difference())))
        assert equal


@pytest.mark.xfail(reason="RuntimeError: cuSOLVER error")  # TODO: create issue
@given(
    dataset=split_datasets(
        standard_regression_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_samples=st.integers(min_value=200, max_value=1000),
            n_features=st.just(20),
            n_informative=st.just(10),
            noise=st.just(20),
        ),
    ),
    algorithm=st.sampled_from(("eig", "svd")),
    fit_intercept=st.booleans(),
    normalize=st.booleans(),
    distribution=st.sampled_from(("lognormal", "exponential", "uniform")),
)
@settings(deadline=5000)
def test_weighted_ridge(dataset, algorithm, fit_intercept, normalize,
                        distribution):
    assume(cuml_compatible_dataset(*dataset))
    assume(sklearn_compatible_dataset(*dataset))

    X_train, X_test, y_train, _ = dataset

    # set weight per sample to be from 1 to max_weight
    if distribution == "uniform":
        wt = np.random.randint(1, high=10, size=len(X_train))
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

    equal = array_equal(skridge_predict, curidge_predict, 1e-1, with_sign=True)
    note(equal)
    target(float(np.abs(equal.compute_difference())))
    assert equal


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_samples=st.integers(min_value=200, max_value=1000),
            n_features=st.just(20),
            n_informative=st.just(10),
            n_classes=st.sampled_from((2, 10)),
        ),
    ),
    penalty=st.sampled_from(("none", "l1", "l2", "elasticnet")),
    l1_ratio=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    fit_intercept=st.booleans(),
    C=st.floats(min_value=0.5, max_value=2.0),
    tol=st.floats(min_value=1e-8, max_value=1e-3),
)
@settings(deadline=5000)
# ignoring UserWarnings in sklearn about setting unused parameters
# like l1 for none penalty
@pytest.mark.xfail(reason="The test is flaky.")
@pytest.mark.filterwarnings("ignore::UserWarning:sklearn[.*]")
def test_logistic_regression(dataset, penalty, l1_ratio, fit_intercept, C,
                             tol):
    # Checking sklearn >= 0.21 for testing elasticnet
    sk_check = LooseVersion(str(sklearn.__version__)) >= LooseVersion("0.21.0")
    if not sk_check and penalty == "elasticnet":
        pytest.skip(
            "Need sklearn > 0.21 for testing logistic with" "elastic net."
        )

    assume(cuml_compatible_dataset(*dataset))
    assume(sklearn_compatible_dataset(*dataset))

    if penalty == 'elasticnet':
        assume(l1_ratio is not None)

    X_train, X_test, y_train, y_test = dataset
    note(f"dataset {dataset}")

    # Avoid a RuntimeError when selecting penalty function that is incompatible
    # with the number of classes.
    if penalty in ("none", "l2"):  # TODO: verify this
        n_classes = len(np.unique(np.concatenate((y_train, y_test))))
        assume(n_classes == 2)  # else fails with RuntimeError

    # Either the train/ or test labels may not contain all labels, unclear
    # whether sklearn makes the same assumption.
    assume(len(np.unique(y_train)) == np.max(y_train) + 1)

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

    assert np.isfinite(X_train).all()
    sklog.fit(X_train, y_train)

    culog_train_score = culog.score(X_train, y_train)
    sklog_train_score = sklog.score(X_train, y_train)
    assume(sklog_train_score > 0)
    assert culog_train_score > 0
    target(
        sklog_train_score - culog_train_score,
        label="Train score diff (skl-cuml)")
    train_score_ratio = culog_train_score / sklog_train_score
    event(f"Train score (cuml/skl): {round(train_score_ratio, 2)}")
    assert train_score_ratio > .945

    culog_test_score = culog.score(X_test, y_test)
    sklog_test_score = sklog.score(X_test, y_test)
    assume(sklog_test_score > 0)
    assert culog_test_score > 0
    target(
        sklog_test_score - culog_test_score,
        label="Test score diff (skl-cuml)")
    test_score_ratio = culog_test_score / sklog_test_score
    event(f"Test score (cuml/skl): {round(test_score_ratio, 2)}")
    assert test_score_ratio > .921

    # Measured results:
    # - Events:
    #   * 75.00%, Test score (cuml/skl): 1.0
    #   * 36.33%, Train score (cuml/skl): 1.0
    #   * 29.69%, Train score (cuml/skl): 0.98
    #   * 10.94%, Train score (cuml/skl): 0.99
    #   * 1.56%, Test score (cuml/skl): 0.98
    #   * 1.17%, Train score (cuml/skl): 1.01
    #   * 0.78%, Test score (cuml/skl): 0.99
    #   * 0.78%, Test score (cuml/skl): 1.01
    #   * 0.39%, Train score (cuml/skl): 1.02

    # - Highest target scores:
    #     0.00956282  (label='Test score diff (skl-cuml)')
    #     0.015748  (label='Train score diff (skl-cuml)')

    all_labels = set(np.unique(np.concatenate((y_train, y_test))))
    sk_preds = sklog.predict(X_test)
    cu_preds = culog.predict(X_test)
    assert all(label in all_labels for label in np.unique(sk_preds))
    assert all(label in all_labels for label in np.unique(cu_preds))


@given(
    dtype=floating_dtypes(sizes=(32, 64)),
    penalty=st.sampled_from(("none", "l1", "l2", "elasticnet")),
    l1_ratio=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
@settings(deadline=5000)
def test_logistic_regression_unscaled(dtype, penalty, l1_ratio):
    if penalty == 'elasticnet':
        assume(l1_ratio is not None)

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

    score_train = culog.score(X_train, y_train)
    score_test = culog.score(X_test, y_test)

    target(1 / score_train, label="inverse train score")
    target(1 / score_test, label="inverse test score")

    assert score_train >= 0.94
    assert score_test >= 0.94


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
        )
    )
)
@pytest.mark.xfail(reason="qn.h: logistic loss invalid C")  # TODO: file issue
@settings(deadline=5000)
def test_logistic_regression_model_default(dataset):
    X_train, X_test, y_train, y_test = dataset

    culog = cuLog()
    culog.fit(X_train, y_train)
    culog_score = culog.score(X_test, y_test)
    target(1 / culog_score, label="inverse culog score")

    sklog = skLog(multi_class="auto")
    sklog.fit(X_train, y_train)
    sklog_score = sklog.score(X_test, y_test)
    target(1 / sklog_score, label="inverse sklog score")

    diff = abs(culog_score - sklog_score)
    target(diff, label="sklog and culog score difference")

    assert culog_score >= sklog_score - 0.022


@given(
    dtype=floating_dtypes(sizes=(32, 64)),
    order=st.sampled_from(("C", "F")),
    sparse_input=st.booleans(),
    fit_intercept=st.booleans(),
    penalty=st.sampled_from(("none", "l1", "l2")),

)
@settings(deadline=5000)
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


@given(dtype=floating_dtypes(sizes=(32, 64)))
@settings(deadline=5000)
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
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@settings(deadline=5000)
def test_logistic_regression_decision_function(
        dataset, fit_intercept, sparse_input):

    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    sklog = skLog(fit_intercept=fit_intercept)
    sklog.coef_ = culog.coef_.T
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0

    cu_dec_func = culog.decision_function(X_test)
    if cu_dec_func.shape[0] > 2:  # num_classes
        cu_dec_func = cu_dec_func.T
    sk_dec_func = sklog.decision_function(X_test)

    assert array_equal(cu_dec_func, sk_dec_func)


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
            n_classes=st.sampled_from((2, 10)),
            n_features=st.just(20),
            n_informative=st.just(10),
        )
    ),
    fit_intercept=st.booleans(),
    sparse_input=st.booleans(),
)
@settings(deadline=5000)
def test_logistic_regression_predict_proba(
        dataset, fit_intercept, sparse_input):

    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))

    if sparse_input:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

    culog = cuLog(fit_intercept=fit_intercept, output_type="numpy")
    culog.fit(X_train, y_train)

    num_classes = len(np.unique(y_train))
    sklog = skLog(
        fit_intercept=fit_intercept,
        **(
            {"solver": "lbfgs", "multi_class": "multinomial"}
            if num_classes > 2
            else {}
        )
    )
    sklog.coef_ = culog.coef_.T
    sklog.intercept_ = culog.intercept_ if fit_intercept else 0
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


@given(
    dataset=split_datasets(
        standard_regression_datasets(
            dtypes=floating_dtypes(sizes=(32, 64))
        )
    ),
    test_dtype=floating_dtypes(sizes=(32, 64)),
)
@settings(deadline=5000)
def test_ridge_predict_convert_dtype(dataset, test_dtype):
    assume(cuml_compatible_dataset(*dataset))
    X_train, X_test, y_train, _ = dataset

    clf = cuRidge()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@given(
    dataset=split_datasets(
        standard_classification_datasets(
            dtypes=floating_dtypes(sizes=(32, 64))
        )
    ),
    test_dtype=floating_dtypes(sizes=(32, 64)),
)
@settings(deadline=5000)
def test_logistic_predict_convert_dtype(dataset, test_dtype):
    X_train, X_test, y_train, y_test = dataset

    # Assumption needed to avoid qn.h: logistic loss invalid C error.
    assume(set(np.unique(y_train)) == set(np.unique(y_test)))

    clf = cuLog()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.xfail(reason="https://github.com/rapidsai/cuml/issues/4963")
@given(
    dataset=standard_classification_datasets(
        dtypes=floating_dtypes(sizes=(32, 64)),
        n_classes=st.integers(2, 7),
        n_features=st.integers(20, 40),
    ),
    sample_weight=st.sampled_from((None, True)),
    class_weight=st.sampled_from((None, "balanced", True)),
)
@settings(deadline=None)
def test_logistic_regression_weighting(dataset, sample_weight, class_weight):
    X, y = dataset

    n_samples = len(X)
    n_classes = len(np.unique(y))

    if sample_weight is True:
        sample_weight = np.abs(np.random.rand(n_samples))
    if class_weight is True:
        class_weight = np.random.rand(n_classes)
        class_weight = {i: c for i, c in enumerate(class_weight)}

    culog = cuLog(fit_intercept=False, class_weight=class_weight)
    culog.fit(X, y, sample_weight=sample_weight)

    sklog = skLog(fit_intercept=False, class_weight=class_weight)
    sklog.fit(X, y, sample_weight=sample_weight)

    # Check coefficients equality.
    unit_tol, total_tol = (.08, .12) if n_classes == 2 else (.3, .4)
    equal = array_equal(
        np.squeeze(culog.coef_),
        np.squeeze(sklog.coef_).T,
        unit_tol=unit_tol, total_tol=total_tol)
    note(equal)
    assert equal

    # Check output probabilities equality.
    equal = array_equal(
        culog.predict_proba(X),
        sklog.predict_proba(X),
        unit_tol=.5, total_tol=.5)
    note(equal)
    assert equal

    # Check output predicted classes equality.
    try:
        # This tests would likely XPASS without this check!
        equal = array_equal(
            culog.predict(X), sklog.predict(X),
            unit_tol=.5, total_tol=.5)
        note(equal)
        assert equal
    except AssertionError:
        pytest.xfail(
            "Probabilities are equal, but predictected classes are not.")


@pytest.mark.xfail(reason="RuntimeError: cuSOLVER error")  # TODO: create issue
@pytest.mark.parametrize('algo', [cuLog, cuRidge])
# ignoring warning about change of solver
@pytest.mark.filterwarnings("ignore::UserWarning:cuml[.*]")
@given(
    dataset=standard_regression_datasets(
        dtypes=floating_dtypes(sizes=(32, 64))
    ),
)
@settings(deadline=None)
def test_linear_models_set_params(algo, dataset):
    x, y = dataset

    model = algo()
    model.fit(x, y)
    coef_before = model.coef_

    if algo == cuLog:
        model = algo(penalty='none', C=1, max_iter=30)
        params = {'penalty': "none", 'C': 1, 'max_iter': 30}
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


@given(
    dataset=split_datasets(
        standard_regression_datasets(
            dtypes=floating_dtypes(sizes=(32, 64)),
        ),
    ),
    alpha=st.floats(.1, 10.0),
    l1_ratio=st.floats(.1, .9),
)
@settings(deadline=None)
def test_elasticnet_solvers_eq(dataset, alpha, l1_ratio):
    assume(cuml_compatible_dataset(* dataset))
    X_train, X_test, y_train, _ = dataset

    kwargs = {'alpha': alpha, 'l1_ratio': l1_ratio}
    cd = cuElasticNet(solver='cd', **kwargs)
    cd.fit(X_train, y_train)
    cd_res = cd.predict(X_test)

    qn = cuElasticNet(solver='qn', **kwargs)
    qn.fit(X_train, y_train)

    # the results of the two models should be close (even if both are bad)
    try:
        assert qn.score(X_test, cd_res) > 0.95
    except AssertionError:
        pytest.xfail("Score too low.")

    # coefficients of the two models should be close
    assert np.corrcoef(cd.coef_, qn.coef_)[0, 1] > 0.98
