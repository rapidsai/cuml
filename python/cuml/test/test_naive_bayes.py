#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cupy as cp

import pytest

from sklearn.metrics import accuracy_score
from cuml.naive_bayes import MultinomialNB
from cuml.naive_bayes import GaussianNB
from cuml.common.input_utils import sparse_scipy_to_cp

from numpy.testing import assert_allclose
from sklearn.naive_bayes import MultinomialNB as skNB
from sklearn.naive_bayes import GaussianNB as skGNB

import math

import numpy as np


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test__multinomial_basic_fit_predict_sparse(x_dtype, y_dtype, nlp_20news):
    """
    Cupy Test
    """

    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    # Priming it seems to lower the end-to-end runtime
    model = MultinomialNB()
    model.fit(X, y)

    cp.cuda.Stream.null.synchronize()

    with cp.prof.time_range(message="start", color_id=10):
        model = MultinomialNB()
        model.fit(X, y)

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.924


def test_gnb():

    from sklearn.utils._testing import assert_array_equal
    from sklearn.utils._testing import assert_array_almost_equal
    from sklearn.utils._testing import assert_raises

    # Data is just 6 separable points in the plane
    X = cp.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]], dtype=cp.float32)
    y = cp.array([1, 1, 1, 2, 2, 2])

    from sklearn.naive_bayes import GaussianNB as GNB

    skclf = GNB()
    skclf.fit(X.get(), y.get())

    print(str(skclf.theta_))
    print(str(skclf.sigma_))

    y_pred_sk = skclf.predict(X.get())

    from cuml.common import logger

    logger.set_level(logger.level_trace)

    clf = GaussianNB(verbose=logger.level_trace)
    clf.fit(X, y)

    print(str(clf.theta_))
    print(str(clf.sigma_))

    y_pred = clf.predict(X)

    print(str(y))
    print(str(y_pred))

    assert_array_equal(y_pred.get(), y.get())

    y_pred_proba = clf.predict_proba(X)

    logger.debug("cuml predict_proba: " + str(y_pred_proba))

    logger.debug("sklearn predict_proba: "+ str(skclf.predict_proba(X.get())))

    y_pred_log_proba = clf.predict_log_proba(X)

    logger.debug(str(y_pred_log_proba))

    logger.debug(str(skclf.predict_log_proba(X.get())))

    assert_array_almost_equal(np.log(y_pred_proba.get()), y_pred_log_proba.get(), 8)

    # # Test whether label mismatch between target y and classes raises
    # # an Error
    # # FIXME Remove this test once the more general partial_fit tests are merged
    # assert_raises(ValueError, GaussianNB().partial_fit, X, y, classes=[0, 1])


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("n_features", [10, 50])
def test_gaussian_basic_fit_predict_sparse(n_features, x_dtype, y_dtype, nlp_20news):
    """
    Cupy Test
    """

    X, y = nlp_20news

    X = scipy_to_cp(X[:, :n_features], x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    model = GaussianNB()
    model.fit(X, y)

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.924


@pytest.mark.parametrize("x_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_sparse_integral_dtype_fails(x_dtype, y_dtype, nlp_20news):
    X, y = nlp_20news

    X = X.astype(x_dtype)
    y = y.astype(y_dtype)

    # Priming it seems to lower the end-to-end runtime
    model = MultinomialNB()

    with pytest.raises(ValueError):
        model.fit(X, y)

    X = X.astype(cp.float32)
    model.fit(X, y)

    X = X.astype(x_dtype)

    with pytest.raises(ValueError):
        model.predict(X)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64,
                                     cp.int32])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_basic_fit_predict_dense_numpy(x_dtype, y_dtype, nlp_20news):
    """
    Cupy Test
    """
    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, cp.float32)
    y = y.astype(y_dtype)

    X = X.tocsr()[0:500].todense()
    y = y[:500]

    model = MultinomialNB()
    model.fit(np.ascontiguousarray(cp.asnumpy(X).astype(x_dtype)), y)

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    accuracy_score(y, y_hat) >= 0.911


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32,
                                     cp.float32, cp.float64])
def test_multinomial_partial_fit(x_dtype, y_dtype, nlp_20news):
    chunk_size = 500

    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()

    model = MultinomialNB()

    classes = np.unique(y)

    total_fit = 0

    for i in range(math.ceil(X.shape[0] / chunk_size)):

        upper = i*chunk_size+chunk_size
        if upper > X.shape[0]:
            upper = -1

        if upper > 0:
            x = X[i*chunk_size:upper]
            y_c = y[i*chunk_size:upper]
        else:
            x = X[i*chunk_size:]
            y_c = y[i*chunk_size:]

        model.partial_fit(x, y_c, classes=classes)

        total_fit += (upper - (i*chunk_size))

        if upper == -1:
            break

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.924


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_predict_proba(x_dtype, y_dtype, nlp_20news):

    X, y = nlp_20news

    cu_X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    cu_y = y.astype(y_dtype)

    cu_X = cu_X.tocsr()

    y = y.get()

    cuml_model = MultinomialNB()
    sk_model = skNB()

    cuml_model.fit(cu_X, cu_y)

    sk_model.fit(X, y)

    cuml_proba = cuml_model.predict_proba(cu_X).get()
    sk_proba = sk_model.predict_proba(X)

    assert_allclose(cuml_proba, sk_proba, atol=1e-6, rtol=1e-2)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_predict_log_proba(x_dtype, y_dtype, nlp_20news):

    X, y = nlp_20news

    cu_X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    cu_y = y.astype(y_dtype)

    cu_X = cu_X.tocsr()

    y = y.get()

    cuml_model = MultinomialNB()
    sk_model = skNB()

    cuml_model.fit(cu_X, cu_y)

    sk_model.fit(X, y)

    cuml_proba = cuml_model.predict_log_proba(cu_X).get()
    sk_proba = sk_model.predict_log_proba(X)

    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_score(x_dtype, y_dtype, nlp_20news):

    X, y = nlp_20news

    cu_X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    cu_y = y.astype(y_dtype)

    cu_X = cu_X.tocsr()

    y = y.get()

    cuml_model = MultinomialNB()
    sk_model = skNB()

    cuml_model.fit(cu_X, cu_y)

    sk_model.fit(X, y)

    cuml_score = cuml_model.score(cu_X, cu_y)
    sk_score = sk_model.score(X, y)

    THRES = 1e-4

    assert sk_score - THRES <= cuml_score <= sk_score + THRES
