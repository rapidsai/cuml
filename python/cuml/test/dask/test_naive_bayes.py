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

import dask

from dask.distributed import Client

from cuml.dask.common import extract_arr_partitions

from cuml.dask.naive_bayes import MultinomialNB

from sklearn.datasets import fetch_20newsgroups


def to_dask_array(client, sp):

    sp = sp.tocsr().astype(cp.float32)

    arr = dask.array.from_array(sp, chunks=sp.shape, asarray=False,
                                fancy=False).persist()

    f = list(map(lambda x: x[1], client.sync(extract_arr_partitions, arr)))

    def conv(x):
        return cp.sparse.csr_matrix(x, dtype=cp.float32)

    f = client.submit(conv, f[0])

    return dask.array.from_delayed(f, shape=sp.shape,
                                   meta=cp.sparse.csr_matrix(cp.zeros(1),
                                                             dtype=cp.float32))


def load_corpus(client):

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)

    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(alternate_sign=False, norm=None)

    xformed = hv.fit_transform(twenty_train.data).astype(cp.float32)
    X = to_dask_array(client, xformed)

    y = dask.array.from_array(twenty_train.target, asarray=False,
                              fancy=False).astype(cp.int32)

    return X, y


def test_basic_fit_predict(cluster):

    client = Client(cluster)

    X, y = load_corpus(client)

    model = MultinomialNB()

    model.fit(X, y)

    y_hat = model.predict(X)

    from sklearn.metrics import accuracy_score

    y_hat = y_hat.compute()
    y = y.compute()

    assert(accuracy_score(y_hat.get(), y) > .97)
