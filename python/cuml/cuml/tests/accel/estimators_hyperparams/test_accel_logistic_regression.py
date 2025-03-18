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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.mark.parametrize(
    "penalty, solver",
    [
        ("l1", "liblinear"),
        ("l1", "saga"),
        ("l2", "lbfgs"),
        ("l2", "liblinear"),
        ("l2", "sag"),
        ("l2", "saga"),
        ("elasticnet", "saga"),
        (None, "lbfgs"),
        (None, "saga"),
    ],
)
def test_logistic_regression_penalty(classification_data, penalty, solver):
    X, y = classification_data
    kwargs = {"penalty": penalty, "solver": solver, "max_iter": 200}
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5  # l1_ratio is required for elasticnet
    clf = LogisticRegression(**kwargs).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("dual", [True, False])
def test_logistic_regression_dual(classification_data, dual):
    X, y = classification_data
    # 'dual' is only applicable when 'penalty' is 'l2' and 'solver' is 'liblinear'
    if dual:
        clf = LogisticRegression(
            penalty="l2", solver="liblinear", dual=dual, max_iter=200
        ).fit(X, y)
    else:
        clf = LogisticRegression(
            penalty="l2", solver="liblinear", dual=dual, max_iter=200
        ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("tol", [1e-2])
def test_logistic_regression_tol(classification_data, tol):
    X, y = classification_data
    clf = LogisticRegression(tol=tol, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("C", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_logistic_regression_C(classification_data, C):
    X, y = classification_data
    clf = LogisticRegression(C=C, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_logistic_regression_fit_intercept(classification_data, fit_intercept):
    X, y = classification_data
    clf = LogisticRegression(fit_intercept=fit_intercept, max_iter=200).fit(
        X, y
    )
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("intercept_scaling", [0.5, 1.0, 2.0])
def test_logistic_regression_intercept_scaling(
    classification_data, intercept_scaling
):
    X, y = classification_data
    # 'intercept_scaling' is only used when solver='liblinear' and fit_intercept=True
    clf = LogisticRegression(
        solver="liblinear",
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        max_iter=200,
    ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_logistic_regression_class_weight(classification_data, class_weight):
    X, y = classification_data
    clf = LogisticRegression(class_weight=class_weight, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


def test_logistic_regression_class_weight_custom(classification_data):
    X, y = classification_data
    class_weights = {0: 1, 1: 2, 2: 1}
    clf = LogisticRegression(class_weight=class_weights, max_iter=200).fit(
        X, y
    )
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize(
    "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
)
def test_logistic_regression_solver(classification_data, solver):
    X, y = classification_data
    clf = LogisticRegression(solver=solver, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("max_iter", [50, 100, 200, 500])
def test_logistic_regression_max_iter(classification_data, max_iter):
    X, y = classification_data
    clf = LogisticRegression(max_iter=max_iter).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize(
    "multi_class, solver",
    [
        ("ovr", "liblinear"),
        ("ovr", "lbfgs"),
        ("multinomial", "lbfgs"),
        ("multinomial", "newton-cg"),
        ("multinomial", "sag"),
        ("multinomial", "saga"),
        ("auto", "lbfgs"),
        ("auto", "liblinear"),
    ],
)
def test_logistic_regression_multi_class(
    classification_data, multi_class, solver
):
    X, y = classification_data
    if solver == "liblinear" and multi_class == "multinomial":
        pytest.skip("liblinear does not support multinomial multi_class")
    clf = LogisticRegression(
        multi_class=multi_class, solver=solver, max_iter=200
    ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("warm_start", [True, False])
def test_logistic_regression_warm_start(classification_data, warm_start):
    X, y = classification_data
    clf = LogisticRegression(warm_start=warm_start, max_iter=200).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_logistic_regression_l1_ratio(classification_data, l1_ratio):
    X, y = classification_data
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, max_iter=200
    ).fit(X, y)
    y_pred = clf.predict(X)
    accuracy_score(y, y_pred)
