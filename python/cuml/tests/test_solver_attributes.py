# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuml import LogisticRegression as cuLog
from cuml.datasets import make_blobs
from cuml.linear_model import ElasticNet as cumlElastic
from cuml.linear_model import Lasso as cumlLasso
from cuml.linear_model import MBSGDClassifier as cumlMBSGClassifier
from cuml.linear_model import MBSGDRegressor as cumlMBSGRegressor


def test_mbsgd_regressor_attributes():
    X, y = make_blobs()
    clf = cumlMBSGRegressor()
    clf.fit(X, y)

    attrs = [
        "dtype",
        "solver_model",
        "coef_",
        "intercept_",
        "l1_ratio",
        "loss",
        "eta0",
        "batch_size",
        "epochs",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_logistic_regression_attributes():
    X, y = make_blobs()
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
    X, y = make_blobs()
    clf = cumlMBSGClassifier()
    clf.fit(X, y)

    attrs = [
        "dtype",
        "solver_model",
        "coef_",
        "intercept_",
        "l1_ratio",
        "eta0",
        "batch_size",
        "fit_intercept",
        "penalty",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_elastic_net_attributes():
    X, y = make_blobs()
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
    X, y = make_blobs()
    clf = cumlLasso()
    clf.fit(X, y)

    attrs = [
        "n_features_in_",
        "coef_",
        "intercept_",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)
