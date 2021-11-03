# Copyright (c) 2021, NVIDIA CORPORATION.
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
import math
import numpy as np
import pytest
import sklearn.datasets as data
import sklearn.model_selection as dsel
from cuml.test.utils import unit_param, quality_param, stress_param

import sklearn.svm as sk
import cuml.svm as cu


SEED = 42
ERROR_TOLERANCE_REL = 0.1
ERROR_TOLERANCE_ABS = 0.01


def good_enough(myscore: float, refscore: float, training_size: int):
    myerr = 1.0 - myscore
    referr = 1.0 - refscore
    # extra discount for uncertainty based on the training data;
    # totally empirical
    c = (2000 + training_size) / (500 + 10 * training_size)
    thresh_rel = referr * (1 + ERROR_TOLERANCE_REL * c)
    thresh_abs = referr + ERROR_TOLERANCE_ABS * c
    good_rel = myerr <= thresh_rel
    good_abs = myerr <= thresh_abs
    assert good_rel or good_abs, \
        f"The model is surely not good enough " \
        f"(cuml error = {myerr} > " \
        f"min(abs threshold = {thresh_abs}; rel threshold = {thresh_rel}))"


def make_regression_dataset(datatype, nrows, ncols):
    ninformative = max(min(ncols, 5), int(math.ceil(ncols / 5)))
    X, y = data.make_regression(
        n_samples=nrows + 1000,
        n_features=ncols,
        random_state=SEED,
        n_informative=ninformative
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    return dsel.train_test_split(X, y, random_state=SEED, train_size=nrows)


def make_classification_dataset(datatype, nrows, ncols, nclasses):
    n_real_features = min(ncols, int(
        max(nclasses * 2, math.ceil(ncols / 10))))
    n_clusters_per_class = min(
        2, max(1, int(2**n_real_features / nclasses))
        )
    n_redundant = min(
        ncols - n_real_features,
        max(2, math.ceil(ncols / 20)))
    try:
        X, y = data.make_classification(
                n_samples=nrows + 1000,
                n_features=ncols,
                random_state=SEED,
                class_sep=1.0,
                n_informative=n_real_features,
                n_clusters_per_class=n_clusters_per_class,
                n_redundant=n_redundant,
                n_classes=nclasses
            )
    except ValueError:
        pytest.skip(
            "Skipping the test for invalid combination of ncols/nclasses")
    X = X.astype(datatype)
    y = y.astype(datatype)
    return dsel.train_test_split(X, y, random_state=SEED, train_size=nrows)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("loss", [
    "epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("dims", [
    unit_param((3, 1)),
    unit_param((100, 1)),
    unit_param((1000, 10)),
    unit_param((100, 100)),
    unit_param((100, 300)),
    quality_param((10000, 10)),
    quality_param((10000, 50)),
    stress_param((1000000, 1000)),
    stress_param((100000, 10000))])
def test_regression_basic(datatype, loss, dims):

    nrows, ncols = dims
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols
    )

    # solving in primal is not supported by sklearn for this loss type.
    skdual = loss == 'epsilon_insensitive'
    skm = sk.LinearSVR(loss=loss, max_iter=10000, dual=skdual)
    cum = cu.LinearSVR(loss=loss, max_iter=10000)

    skm.fit(X_train, y_train)
    cum.fit(X_train, y_train)

    good_enough(cum.score(X_test, y_test), skm.score(X_test, y_test), nrows)


@pytest.mark.parametrize("loss", [
    "epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("epsilon", [0, 0.001, 0.1])
@pytest.mark.parametrize("dims", [
    quality_param((10000, 10)),
    quality_param((10000, 50)),
    quality_param((10000, 500))])
def test_regression_eps(loss, epsilon, dims):

    nrows, ncols = dims
    X_train, X_test, y_train, y_test = make_regression_dataset(
        np.float32, nrows, ncols
    )

    # solving in primal is not supported by sklearn for this loss type.
    skdual = loss == 'epsilon_insensitive'
    skm = sk.LinearSVR(loss=loss, epsilon=epsilon, max_iter=10000, dual=skdual)
    cum = cu.LinearSVR(loss=loss, epsilon=epsilon, max_iter=10000)

    skm.fit(X_train, y_train)
    cum.fit(X_train, y_train)

    good_enough(cum.score(X_test, y_test), skm.score(X_test, y_test), nrows)

@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("penalty", [
    "l1", "l2"])
@pytest.mark.parametrize("loss", [
    "hinge", "squared_hinge"])
@pytest.mark.parametrize("dims", [
    unit_param((3, 1)),
    unit_param((100, 1)),
    unit_param((1000, 10)),
    unit_param((100, 100)),
    unit_param((100, 300)),
    quality_param((10000, 10)),
    quality_param((10000, 50)),
    stress_param((1000000, 1000)),
    stress_param((100000, 10000))])
@pytest.mark.parametrize("nclasses", [2, 3, 5, 8])
def test_classification(datatype, penalty, loss, dims, nclasses):

    nrows, ncols = dims
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype, nrows, ncols, nclasses
    )

    # solving in primal is not supported by sklearn for this loss type.
    skdual = loss == 'hinge' and penalty == 'l2'
    if loss == 'hinge' and penalty == 'l1':
        pytest.skip(
            "sklearn does not support this combination of loss and penalty")

    skm = sk.LinearSVC(loss=loss, penalty=penalty, max_iter=10000, dual=skdual)
    cum = cu.LinearSVC(loss=loss, penalty=penalty, max_iter=10000)

    skm.fit(X_train, y_train)
    cum.fit(X_train, y_train)

    good_enough(cum.score(X_test, y_test), skm.score(X_test, y_test), nrows)
