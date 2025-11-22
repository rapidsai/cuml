# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuml import LogisticRegression as cuLog
from cuml.datasets import make_classification, make_regression
from cuml.linear_model import ElasticNet as cumlElastic
from cuml.linear_model import Lasso as cumlLasso
from cuml.linear_model import MBSGDClassifier as cumlMBSGClassifier
from cuml.linear_model import MBSGDRegressor as cumlMBSGRegressor


def test_mbsgd_regressor_attributes():
    X, y = make_regression()
    clf = cumlMBSGRegressor()
    clf.fit(X, y)

    attrs = [
        "coef_",
        "intercept_",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_logistic_regression_attributes():
    X, y = make_classification()
    clf = cuLog().fit(X, y, convert_dtype=True)

    attrs = [
        "coef_",
        "intercept_",
        "n_iter_",
        "n_features_in_",
    ]

    for attr in attrs:
        assert hasattr(clf, attr)


def test_mbsgd_classifier_attributes():
    X, y = make_classification()
    clf = cumlMBSGClassifier()
    clf.fit(X, y)

    attrs = [
        "coef_",
        "intercept_",
        "classes_",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_elastic_net_attributes():
    X, y = make_regression()
    clf = cumlElastic(fit_intercept=False)
    clf.fit(X, y)

    attrs = [
        "n_features_in_",
        "coef_",
        "intercept_",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_lasso_attributes():
    X, y = make_regression()
    clf = cumlLasso()
    clf.fit(X, y)

    attrs = [
        "n_features_in_",
        "coef_",
        "intercept_",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)
