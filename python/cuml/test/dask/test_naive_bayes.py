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

import pandas as pd

from sklearn.datasets import fetch_20newsgroups


def scipy_to_cp(sp):
    coo = sp.tocoo()
    values = coo.data

    r = cp.asarray(coo.row)
    c = cp.asarray(coo.col)
    v = cp.asarray(values, dtype=cp.float32)

    return cp.sparse.coo_matrix((v, (r, c)))


def load_corpus():

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=42)

    X = pd.DataFrame(twenty_train.data)
    y = pd.DataFrame(twenty_train.target)

    X = dask.dataframe.from_pandas(X, npartitions=1)
    y = dask.dataframe.from_pandas(y, npartitions=1)

    return X, y


def test_basic_fit_predict(cluster):

    X, y = load_corpus()

    print("before: " + str(X.shape))
    print(str(y.shape))
