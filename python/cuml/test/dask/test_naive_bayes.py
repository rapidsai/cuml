#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import dask

from dask.distributed import Client

from sklearn.feature_extraction.text import HashingVectorizer

from cuml.dask.common import to_sp_dask_array

from sklearn.metrics import accuracy_score

from cuml.dask.naive_bayes import MultinomialNB

from sklearn.datasets import fetch_20newsgroups


def load_corpus(client):

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)

    hv = HashingVectorizer(alternate_sign=False, norm=None)

    xformed = hv.fit_transform(twenty_train.data).astype(cp.float32)

    X = to_sp_dask_array(xformed, client)

    y = dask.array.from_array(twenty_train.target, asarray=False,
                              fancy=False).astype(cp.int32)

    return X, y


def test_basic_fit_predict(cluster):

    client = Client(cluster)

    try:

        X, y = load_corpus(client)

        model = MultinomialNB()

        model.fit(X, y)

        y_hat = model.predict(X)

        y_hat = y_hat.compute()
        y = y.compute()

        assert(accuracy_score(y_hat.get(), y) > .97)
    finally:
        client.close()


def test_score(cluster):

    client = Client(cluster)

    try:
        X, y = load_corpus(client)

        model = MultinomialNB()

        model.fit(X, y)

        y_hat = model.predict(X)

        score = model.score(X, y)

        y_hat = y_hat.compute()
        y = y.compute()

        assert(accuracy_score(y_hat.get(), y) == score)
    finally:
        client.close()
