# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import numpy as np
import pytest
import sklearn.svm
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import cuml
import cuml.svm as cu
from cuml.common.exceptions import NotFittedError
from cuml.testing.utils import as_type


def make_regression_dataset(n_rows, n_cols, dtype=np.float32):
    n_informative = max(min(n_cols, 5), int(math.ceil(n_cols / 5)))
    X, y = make_regression(
        n_samples=n_rows + 1000,
        n_features=n_cols,
        random_state=42,
        n_informative=n_informative,
    )
    X = X.astype(dtype, copy=False)
    y = y.astype(dtype, copy=False)
    return train_test_split(X, y, random_state=42, train_size=n_rows)


def make_classification_dataset(n_rows, n_cols, n_classes, dtype=np.float32):
    if n_cols < n_classes:
        n_informative = n_cols
    else:
        n_informative = min(n_classes, 10)
    n_clusters_per_class = max(math.floor(2**n_informative / n_classes), 1)
    X, y = make_classification(
        n_samples=n_rows + 1000,
        n_features=n_cols,
        random_state=42,
        n_informative=n_informative,
        n_clusters_per_class=n_clusters_per_class,
        n_redundant=0,
        n_classes=n_classes,
    )
    X = X.astype(dtype)
    return train_test_split(
        X, y, random_state=42, train_size=n_rows, stratify=y
    )


def test_linear_svr_input_constraints():
    X, y = make_regression(random_state=42)

    with pytest.raises(ValueError, match=r"0 sample\(s\)"):
        cuml.LinearSVR().fit(X[:0], y[:0])

    with pytest.raises(ValueError, match=r"0 feature\(s\)"):
        cuml.LinearSVR().fit(X[:, :0], y)


def test_linear_svc_input_constraints():
    X, y = make_classification(random_state=42)

    with pytest.raises(ValueError, match=r"0 sample\(s\)"):
        cuml.LinearSVC().fit(X[:0], y[:0])

    with pytest.raises(ValueError, match=r"0 feature\(s\)"):
        cuml.LinearSVC().fit(X[:, :0], y)

    with pytest.raises(ValueError, match="only 1 class"):
        cuml.LinearSVC().fit(X, np.ones_like(y))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]
)
@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (100, 1),
        (1000, 10),
        (100, 100),
        (100, 300),
    ],
)
@pytest.mark.parametrize("epsilon", [0.0, 0.5])
def test_linear_svr(dtype, loss, epsilon, n_rows, n_cols):
    X_train, X_test, y_train, y_test = make_regression_dataset(
        n_rows, n_cols, dtype
    )

    cu_model = cuml.LinearSVR(loss=loss, epsilon=epsilon)
    cu_model.fit(X_train, y_train)
    cu_score = cu_model.score(X_test, y_test)

    sk_model = sklearn.svm.LinearSVR(loss=loss, epsilon=epsilon)
    sk_model.fit(X_train, y_train)
    sk_score = sk_model.score(X_test, y_test)

    assert cu_score >= sk_score - 0.05


def check_linear_svc(
    dtype, penalty, loss, n_rows, n_cols, n_classes, class_weight
):
    X_train, X_test, y_train, y_test = make_classification_dataset(
        n_rows, n_cols, n_classes, dtype=dtype
    )

    cu_model = cuml.LinearSVC(
        loss=loss,
        penalty=penalty,
        class_weight=class_weight,
    )
    cu_model.fit(X_train, y_train)
    cu_score = cu_model.score(X_test, y_test)

    dual = not (penalty == "l1" and loss == "squared_hinge")

    sk_model = sklearn.svm.LinearSVC(
        loss=loss,
        penalty=penalty,
        class_weight=class_weight,
        dual=dual,
    )
    sk_model.fit(X_train, y_train)
    sk_score = sk_model.score(X_test, y_test)

    assert cu_score >= sk_score - 0.05


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "penalty, loss",
    [
        # sklearn doesn't support l1, hinge, skip it
        ("l2", "hinge"),
        ("l1", "squared_hinge"),
        ("l2", "squared_hinge"),
    ],
)
@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (100, 20),
        (1000, 20),
        (100, 150),
    ],
)
def test_linear_svc_binary(dtype, penalty, loss, n_rows, n_cols):
    check_linear_svc(dtype, penalty, loss, n_rows, n_cols, 2, None)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (100, 20),
        (1000, 20),
        (100, 150),
    ],
)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_linear_svc_class_weight_balanced(dtype, n_rows, n_cols, n_classes):
    check_linear_svc(
        dtype, "l2", "hinge", n_rows, n_cols, n_classes, "balanced"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n_rows, n_cols",
    [
        (100, 20),
        (1000, 20),
        (100, 150),
    ],
)
def test_svc_class_weight(dtype, n_rows, n_cols):
    check_linear_svc(dtype, "l2", "hinge", n_rows, n_cols, 2, {0: 0.5, 1: 1.5})


@pytest.mark.parametrize(
    "n_rows, n_cols, n_classes",
    [
        (3, 10, 2),
        (100, 20, 4),
    ],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linear_svc_decision_function(
    n_rows, n_cols, n_classes, fit_intercept
):
    # The decision function is not stable to compare given random
    # input data and models that are similar but not equal.
    # This test will only check the cuml decision function
    # implementation based on an imported model from sklearn.
    X_train, X_test, y_train, y_test = make_classification_dataset(
        n_rows, n_cols, n_classes, dtype=np.float64
    )
    sk_model = sklearn.svm.LinearSVC(
        max_iter=10, dual=False, fit_intercept=fit_intercept
    )
    sk_model.fit(X_train, y_train)
    sol = sk_model.decision_function(X_test)

    cu_model = cu.LinearSVC.from_sklearn(sk_model)
    res = cu_model.decision_function(X_test)

    np.testing.assert_allclose(res, sol, atol=1e-4)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_linear_svc_predict_proba(fit_intercept, n_classes):
    n_rows, n_cols = 500, 20
    X_train, X_test, y_train, y_test = make_classification_dataset(
        n_rows, n_cols, n_classes
    )

    cu_model = cuml.LinearSVC(fit_intercept=fit_intercept, probability=True)
    cu_model.fit(X_train, y_train)
    cu_score = cu_model.score(X_test, y_test)

    sk_model = sklearn.svm.LinearSVC(fit_intercept=fit_intercept)
    sk_model.fit(X_train, y_train)
    sk_score = sk_model.score(X_test, y_test)

    assert cu_score >= sk_score - 0.05

    proba = cu_model.predict_proba(X_test)
    log_proba = cu_model.predict_log_proba(X_test)

    # log_proba is log(proba)
    np.testing.assert_allclose(np.log(proba), log_proba, rtol=1e-4)

    # Probabilities sum to 1
    np.testing.assert_allclose(
        proba.sum(axis=1), np.ones(len(proba)), rtol=1e-4
    )

    # Predictions are argmax(proba)
    y_pred = cu_model.predict(X_test)
    y_pred_proba = cu_model.classes_.take(proba.argmax(axis=1).astype("int32"))
    np.testing.assert_array_equal(y_pred, y_pred_proba)


def test_linear_svc_predict_proba_not_available():
    X_train, X_test, y_train, y_test = make_classification_dataset(100, 20, 2)
    model = cuml.LinearSVC().fit(X_train, y_train)

    with pytest.raises(NotFittedError, match="probability=True"):
        model.predict_proba(X_test)

    with pytest.raises(NotFittedError, match="probability=True"):
        model.predict_log_proba(X_test)


@pytest.mark.parametrize("kind", ["numpy", "pandas", "cupy", "cudf"])
@pytest.mark.parametrize("weighted", [False, True])
def test_linear_svc_input_types(kind, weighted):
    X, y = make_classification()
    if weighted:
        sample_weight = np.random.default_rng(42).random(X.shape[0])
    else:
        sample_weight = None
    X, y, sample_weight = as_type(kind, X, y, sample_weight)
    model = cuml.LinearSVC()
    model.fit(X, y, sample_weight=sample_weight)
    y_pred = model.predict(X)
    # predict output type matches input type
    assert type(y_pred).__module__.split(".")[0] == kind
    assert y_pred.dtype == y.dtype
