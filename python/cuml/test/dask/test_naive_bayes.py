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


from dask.distributed import Client
from cuml.test.dask.utils import load_text_corpus

from sklearn.metrics import accuracy_score

from cuml.dask.naive_bayes import MultinomialNB
from cuml.naive_bayes.naive_bayes import MultinomialNB as SGNB


def test_basic_fit_predict(cluster):

    client = Client(cluster)

    try:

        X, y = load_text_corpus(client)

        model = MultinomialNB()

        model.fit(X, y)

        y_hat = model.predict(X)

        y_hat = y_hat.compute()
        y = y.compute()

        assert(accuracy_score(y_hat.get(), y) > .97)
    finally:
        client.close()


def test_single_distributed_exact_results(cluster):

    client = Client(cluster)

    try:

        X, y = load_text_corpus(client)

        sgX, sgy = (X.compute(), y.compute())

        model = MultinomialNB()
        model.fit(X, y)

        sg_model = SGNB()
        sg_model.fit(sgX, sgy)

        y_hat = model.predict(X)
        sg_y_hat = sg_model.predict(sgX).get()

        y_hat = y_hat.compute().get()

        assert(accuracy_score(y_hat, sg_y_hat) == 1.0)
    finally:
        client.close()


def test_score(cluster):

    client = Client(cluster)

    try:
        X, y = load_text_corpus(client)

        model = MultinomialNB()
        model.fit(X, y)

        y_hat = model.predict(X)

        score = model.score(X, y)

        y_hat_local = y_hat.compute()
        y_local = y.compute()

        assert(accuracy_score(y_hat_local.get(), y_local) == score)
    finally:
        client.close()
