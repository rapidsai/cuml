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
from cuml.naive_bayes import BernoulliNB
from cuml.naive_bayes import CategoricalNB
from cuml.common.input_utils import sparse_scipy_to_cp
from cuml.datasets import make_classification

from numpy.testing import assert_allclose, assert_array_equal
from sklearn.naive_bayes import MultinomialNB as skNB
from sklearn.naive_bayes import GaussianNB as skGNB
from sklearn.naive_bayes import CategoricalNB as skCNB
from sklearn.naive_bayes import BernoulliNB as skBNB

import math

import numpy as np


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_multinomial_basic_fit_predict_sparse(x_dtype, y_dtype, nlp_20news):
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

    #with cp.prof.time_range(message="start", color_id=10):
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
    X = cp.array([[-2, -1, -1], [-1, -1, -1], [-1, -2, -1],
                  [1, 1, 1], [1, 2, 1], [2, 1, 1]], dtype=cp.float32)
    y = cp.array([1, 1, 1, 2, 2, 2])

    skclf = skGNB()
    skclf.fit(X.get(), y.get())

    y_pred_sk = skclf.predict(X.get())

    from cuml.common import logger

    logger.set_level(logger.level_trace)

    clf = GaussianNB(verbose=logger.level_trace)
    clf.fit(X, y)

    assert_array_almost_equal(clf.theta_.get(), skclf.theta_, 6)
    assert_array_almost_equal(clf.sigma_.get(), skclf.sigma_, 6)


    y_pred = clf.predict(X)

    assert_array_equal(y_pred.get(), y.get())

    y_pred_proba = clf.predict_proba(X)
    y_pred_proba_sk = skclf.predict_proba(X.get())


    y_pred_log_proba = clf.predict_log_proba(X)
    y_pred_log_proba_sk = skclf.predict_log_proba(X.get())

    assert_array_almost_equal(y_pred_proba.get(), y_pred_proba_sk, 8)
    assert_allclose(y_pred_log_proba.get(), y_pred_log_proba_sk, atol=1e-2, rtol=1e-2)

    # # Test whether label mismatch between target y and classes raises
    # # an Error
    # # FIXME Remove this test once the more general partial_fit tests are merged
    # assert_raises(ValueError, GaussianNB().partial_fit, X, y, classes=[0, 1])


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
def test_gaussian_basic_fit_predict(x_dtype, y_dtype, is_sparse,
                                    nlp_20news):
    """
    Cupy Test
    """

    X, y = nlp_20news
    model = GaussianNB()

    if is_sparse:
        X = sparse_scipy_to_cp(X[:1000], x_dtype).astype(x_dtype)
        y = y.astype(y_dtype)[:1000]
        model.fit(X, y)
    else:
        X = sparse_scipy_to_cp(X, x_dtype)
        X = X.tocsr()[0:1000].todense()
        y = y[0:1000].astype(y_dtype)
        model.fit(np.ascontiguousarray(cp.asnumpy(X).astype(x_dtype)), y)

    y_hat = model.predict(X)
    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.924


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32,
                                     cp.float32, cp.float64])
def test_gaussian_partial_fit(x_dtype, y_dtype, nlp_20news):
    chunk_size = 500

    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()

    model = GaussianNB()
    modelsk = skGNB()

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

        modelsk.partial_fit(x.get().toarray(), y_c.get(), classes=classes.get())
        model.partial_fit(x, y_c, classes=classes)

        total_fit += (upper - (i*chunk_size))

        if upper == -1:
            break

    y_hat = model.predict(X)
    y_sk = modelsk.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)
    assert_array_equal(y_hat, y_sk)

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

    assert accuracy_score(y, y_hat) >= 0.81


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


@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("is_sparse", [True, False])
def test_bernoulli(x_dtype, y_dtype, is_sparse, nlp_20news):
    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()[0:500]
    y = y[:500]
    if not is_sparse:
        X = X.todense()

    sk_model = skBNB()
    cuml_model = BernoulliNB()

    sk_model.fit(X.get(), y.get())
    cuml_model.fit(X, y)

    sk_score = sk_model.score(X.get(), y.get())
    cuml_score = cuml_model.score(X, y)
    cuml_proba = cuml_model.predict_log_proba(X).get()
    sk_proba = sk_model.predict_log_proba(X.get())

    THRES = 1e-3

    assert_array_equal(sk_model.class_count_, cuml_model.class_count_.get())
    assert_allclose(sk_model.class_log_prior_, cuml_model.class_log_prior_.get(), 1e-6)
    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES



@pytest.mark.parametrize("x_dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("y_dtype", [cp.int32,
                                     cp.float32, cp.float64])
def test_bernoulli_partial_fit(x_dtype, y_dtype, nlp_20news):
    chunk_size = 500

    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, x_dtype).astype(x_dtype)
    y = y.astype(y_dtype)

    X = X.tocsr()

    model = BernoulliNB()

    classes = np.unique(y)

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
        if upper == -1:
            break

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    assert accuracy_score(y, y_hat) >= 0.78



@pytest.mark.parametrize("x_dtype", [cp.int32,
                                     cp.float64, cp.float32])
@pytest.mark.parametrize("y_dtype", [cp.int32, cp.int64])
def test_categorical(x_dtype, y_dtype, nlp_20news):
    X, y = nlp_20news

    X = sparse_scipy_to_cp(X, cp.float32)
    y = y.astype(y_dtype)

    X = X.tocsr()[0:500, :5000].todense().astype(x_dtype)
    y = y[:500]

    sk_model = skCNB()
    sk_model.fit(X.get(), y.get())

    cuml_model = CategoricalNB()
    cuml_model.fit(X, y)

    sk_score = sk_model.score(X.get(), y.get())
    cuml_score = cuml_model.score(X, y)
    cuml_proba = cuml_model.predict_log_proba(X).get()
    sk_proba = sk_model.predict_log_proba(X.get())

    THRES = 1e-3

    assert_array_equal(sk_model.class_count_, cuml_model.class_count_.get())
    assert_allclose(sk_model.class_log_prior_, cuml_model.class_log_prior_.get(), 1e-6)
    assert_allclose(cuml_proba, sk_proba, atol=1e-2, rtol=1e-2)
    assert sk_score - THRES <= cuml_score <= sk_score + THRES

