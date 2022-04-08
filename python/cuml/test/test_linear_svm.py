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

import gc
import multiprocessing as mp
import time
import math
import cupy as cp
import numpy as np
import pytest
import cuml.datasets as data
import cuml.model_selection as dsel
from queue import Empty
from cuml.test.utils import unit_param, quality_param, stress_param

import sklearn.svm as sk
import cuml.svm as cu
import cuml
import cuml.common.logger as logger


SEED = 42
ERROR_TOLERANCE_REL = 0.1
ERROR_TOLERANCE_ABS = 0.01
SKLEARN_TIMEOUT_FACTOR = 10


def good_enough(myscore: float, refscore: float, training_size: int):
    myerr = 1.0 - myscore
    referr = 1.0 - refscore
    # Extra discount for uncertainty based on the training data.
    # Totally empirical; for <10 samples, the error is allowed
    # to be ~50%, which is a total randomness. But this is ok,
    # since we don't expect the model to be trained from this few
    # samples.
    c = (10000 + training_size) / (100 + 5 * training_size)
    thresh_rel = referr * (1 + ERROR_TOLERANCE_REL * c)
    thresh_abs = referr + ERROR_TOLERANCE_ABS * c
    good_rel = myerr <= thresh_rel
    good_abs = myerr <= thresh_abs
    assert good_rel or good_abs, \
        f"The model is surely not good enough " \
        f"(cuml error = {myerr} > " \
        f"min(abs threshold = {thresh_abs}; rel threshold = {thresh_rel}))"


def with_timeout(timeout, target, args=(), kwargs={}):
    '''Don't wait if the sklearn function takes really too long.'''
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        logger.warn(
            '"fork" multiprocessing start method is not available. '
            'The sklearn model will run in the same process and '
            'cannot be killed if it runs too long.')
        return target(*args, **kwargs)
    q = ctx.Queue()

    def target_res():
        try:
            q.put((True, target(*args, **kwargs)))
        except BaseException as e:  # noqa E722
            print("Test subprocess failed with an exception: ", e)
            q.put((False, None))
    p = ctx.Process(target=target_res)
    p.start()
    try:
        success, val = q.get(True, timeout)
        if success:
            return val
        else:
            raise RuntimeError("Got an exception in the subprocess.")
    except Empty:
        p.terminate()
        raise TimeoutError()


def make_regression_dataset(datatype, nrows, ncols):
    ninformative = max(min(ncols, 5), int(math.ceil(ncols / 5)))
    X, y = data.make_regression(
        dtype=datatype,
        n_samples=nrows + 1000,
        n_features=ncols,
        random_state=SEED,
        n_informative=ninformative
    )
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
                dtype=datatype,
                n_samples=nrows + 1000,
                n_features=ncols,
                random_state=SEED,
                class_sep=1.0,
                n_informative=n_real_features,
                n_clusters_per_class=n_clusters_per_class,
                n_redundant=n_redundant,
                n_classes=nclasses
            )

        r = dsel.train_test_split(X, y, random_state=SEED, train_size=nrows)

        if len(cp.unique(r[2])) < nclasses:
            raise ValueError("Training data does not have all classes.")

        return r

    except ValueError:
        pytest.skip(
            "Skipping the test for invalid combination of ncols/nclasses")


def run_regression(datatype, loss, eps, dims):

    nrows, ncols = dims
    X_train, X_test, y_train, y_test = make_regression_dataset(
        datatype, nrows, ncols
    )

    # solving in primal is not supported by sklearn for this loss type.
    skdual = loss == 'epsilon_insensitive'
    # limit the max iterations for sklearn to reduce the max test time
    cuit = 10000
    skit = max(10, min(cuit, cuit * 1000 / nrows))

    t = time.perf_counter()
    cum = cu.LinearSVR(loss=loss, epsilon=eps, max_iter=cuit)
    cum.fit(X_train, y_train)
    cus = cum.score(X_test, y_test)
    t = max(5, (time.perf_counter() - t) * SKLEARN_TIMEOUT_FACTOR)

    # cleanup cuml objects so that we can more easily fork the process
    # and test sklearn
    del cum
    X_train = X_train.get()
    X_test = X_test.get()
    y_train = y_train.get()
    y_test = y_test.get()
    gc.collect()

    try:
        def run_sklearn():
            skm = sk.LinearSVR(
                loss=loss, epsilon=eps, max_iter=skit, dual=skdual)
            skm.fit(X_train, y_train)
            return skm.score(X_test, y_test)
        sks = with_timeout(timeout=t, target=run_sklearn)
        good_enough(cus, sks, nrows)
    except TimeoutError:
        pytest.skip(f"sklearn did not finish within {t} seconds.")


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
    stress_param((100000, 1000))])
def test_regression_basic(datatype, loss, dims):
    run_regression(datatype, loss, 0, dims)


@pytest.mark.parametrize("loss", [
    "epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("epsilon", [0, 0.001, 0.1])
@pytest.mark.parametrize("dims", [
    quality_param((10000, 10)),
    quality_param((10000, 50)),
    quality_param((10000, 500))])
def test_regression_eps(loss, epsilon, dims):
    run_regression(np.float32, loss, epsilon, dims)


def run_classification(datatype, penalty, loss, dims, nclasses):

    t = time.perf_counter()
    nrows, ncols = dims
    X_train, X_test, y_train, y_test = make_classification_dataset(
        datatype, nrows, ncols, nclasses
    )
    logger.debug(f"Data generation time: {time.perf_counter() - t} s.")

    # solving in primal is not supported by sklearn for this loss type.
    skdual = loss == 'hinge' and penalty == 'l2'
    if loss == 'hinge' and penalty == 'l1':
        pytest.skip(
            "sklearn does not support this combination of loss and penalty")

    # limit the max iterations for sklearn to reduce the max test time
    cuit = 10000
    skit = max(10, min(cuit, cuit * 1000 / nrows))

    t = time.perf_counter()
    handle = cuml.Handle(n_streams=0)
    cum = cu.LinearSVC(
        handle=handle, loss=loss, penalty=penalty, max_iter=cuit)
    cum.fit(X_train, y_train)
    cus = cum.score(X_test, y_test)
    handle.sync()
    t = time.perf_counter() - t
    logger.debug(f"Cuml time: {t} s.")
    t = max(5, t * SKLEARN_TIMEOUT_FACTOR)

    # cleanup cuml objects so that we can more easily fork the process
    # and test sklearn
    del cum
    X_train = X_train.get()
    X_test = X_test.get()
    y_train = y_train.get()
    y_test = y_test.get()
    gc.collect()

    try:
        def run_sklearn():
            skm = sk.LinearSVC(
                loss=loss, penalty=penalty, max_iter=skit, dual=skdual)
            skm.fit(X_train, y_train)
            return skm.score(X_test, y_test)
        sks = with_timeout(timeout=t, target=run_sklearn)
        good_enough(cus, sks, nrows)
    except TimeoutError:
        pytest.skip(f"sklearn did not finish within {t} seconds.")


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
    stress_param((100000, 1000))])
def test_classification_1(datatype, penalty, loss, dims):
    run_classification(datatype, penalty, loss, dims, 2)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("dims", [
    unit_param((3, 1)),
    unit_param((100, 1)),
    unit_param((1000, 10)),
    unit_param((100, 100)),
    unit_param((100, 300)),
    quality_param((10000, 10)),
    quality_param((10000, 50)),
    stress_param((100000, 1000))])
@pytest.mark.parametrize("nclasses", [2, 3, 5, 8])
def test_classification_2(datatype, dims, nclasses):
    run_classification(datatype, "l2", "hinge", dims, nclasses)
