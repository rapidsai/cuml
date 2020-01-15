#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from cuml.naive_bayes import MultinomialNB

import math

import numpy as np


def scipy_to_cp(sp):
    coo = sp.tocoo()
    values = coo.data

    r = cp.asarray(coo.row)
    c = cp.asarray(coo.col)
    v = cp.asarray(values, dtype=cp.float32)

    return cp.sparse.coo_matrix((v, (r, c)))


def load_corpus():

    twenty_train = fetch_20newsgroups(subset='train',
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    X = scipy_to_cp(X)

    return X, Y


def test_basic_fit_predict_sparse():
    """
    Cupy Test
    """

    X, y = load_corpus()

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


def test_basic_fit_predict_dense():
    """
    Cupy Test
    """
    X, y = load_corpus()

    X = X.tocsr()[0:5000].todense()
    y = y[:5000]

    model = MultinomialNB()
    model.fit(np.ascontiguousarray(cp.asnumpy(X)), y)

    y_hat = model.predict(X)

    y_hat = cp.asnumpy(y_hat)
    y = cp.asnumpy(y)

    accuracy_score(y, y_hat) >= 0.911


def test_partial_fit():
    chunk_size = 500

    X, y = load_corpus()
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
