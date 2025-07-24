# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
        "dtype",
        "solver_model",
        "coef_",
        "intercept_",
        "l1_ratio",
        "C",
        "penalty",
        "fit_intercept",
        "solver",
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
        "dtype",
        "solver_model",
        "coef_",
        "intercept_",
        "l1_ratio",
        "alpha",
        "max_iter",
        "fit_intercept",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)


def test_lasso_attributes():
    X, y = make_blobs()
    clf = cumlLasso()
    clf.fit(X, y)

    attrs = [
        "dtype",
        "solver_model",
        "coef_",
        "intercept_",
        "solver_model",
        "l1_ratio",
    ]
    for attr in attrs:
        assert hasattr(clf, attr)
