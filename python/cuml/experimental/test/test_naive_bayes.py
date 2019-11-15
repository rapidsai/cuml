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

import torch

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from cuml.experimental.naive_bayes import MultinomialNB


def scipy_to_torch(sp):
    coo = sp.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.cuda.LongTensor(indices)
    v = torch.cuda.FloatTensor(values)

    return torch.cuda.sparse.FloatTensor(i, v, torch.Size(coo.shape))


def load_corpus():

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = torch.cuda.IntTensor(twenty_train.target).cuda()

    return scipy_to_torch(X), Y


def test_basic_fit_predict():

    X, y = load_corpus()

    model = MultinomialNB()
    model.fit(X, y)

    y_hat = model.predict(X).cpu()

    assert accuracy_score(y.cpu(), y_hat) >= 0.996



