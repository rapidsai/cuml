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

import time

import math

import torch

import numpy as np


def scipy_to_torch(sp):
    coo = sp.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.cuda.LongTensor(indices)
    v = torch.cuda.FloatTensor(values)

    return torch.cuda.sparse.FloatTensor(i, v, torch.Size(coo.shape))


class PyTorchBayes(object):

    def __init__(self, l, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.l = l
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.n_features_ = None

    def fit(self, X, y, _partial=False, _classes=None):

        self.n_features_ = X.shape[1]
        self._init_counters(y.shape[1], X.shape[1])

        self.classes_ = self.l.classes_
        self.n_classes_ = self.l.classes_.shape[0]

        self._count(X, y)
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def predict(self, X):
        jll = self._joint_log_likelihood(X)

        _, indices = torch.max(jll, 1)
        return indices

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = torch.zeros(n_effective_classes).cuda()
        self.feature_count_ = torch.zeros(n_effective_classes, n_features).cuda()

    def _count(self, X, Y):
        self.feature_count_ += torch.sparse.mm(X.t(), Y).t()
        self.class_count_ += Y.sum(axis=0)

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:
            self.class_log_prior_ = torch.log(class_prior)

        elif self.fit_prior:
            log_class_count = torch.log(self.class_count_)

        self.class_log_prior_ = torch.full((self.n_classes_, 1),
                                           -math.log(self.n_classes_)).cuda()

    def _update_feature_log_prob(self, alpha):
        """ apply add-lambda smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = (torch.log(smoothed_fc) - torch.log(smoothed_cc))

    def _joint_log_likelihood(self, X):
        """ Calculate the posterior log probability of the samples X """
        ret = torch.sparse.mm(X, self.feature_log_prob_.T)
        ret += self.class_log_prior_.T
        return ret


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
    twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = cp.array(twenty_train.target)

    return scipy_to_cp(X), Y


def load_corpus_cpu():

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                      shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = twenty_train.target

    return X, Y


def test_basic_fit_predict():

    """
    Cupy Test
    """

    X, y = load_corpus()

    from cuml.preprocessing import LabelBinarizer as LB

    l = LB()
    y_sparse = l.fit_transform(y)

    print("CLASSES: "+ str(l.classes_))

    # Priming it seems to lower the end-to-end runtime
    model = MultinomialNB()
    model.fit(X, y_sparse, classes=l.classes_, _sparse_labels=True)

    print(str(X.shape))

    start = time.time()
    model = MultinomialNB()
    model.fit(X, y_sparse, classes=l.classes_, _sparse_labels=True)
    end = time.time() - start
    print(str(end))

    y_hat = cp.asnumpy(model.predict(X))
    y = cp.asnumpy(y)

    print(str(y_hat))

    print(str(accuracy_score(y, y_hat)))

    """
    Sklearn Test
    """

    from sklearn.naive_bayes import MultinomialNB as MNB

    X, y = load_corpus_cpu()
    start = time.time()
    model = MNB()
    model.fit(X, y)
    end = time.time() - start
    print(str(end))

    y_hat = model.predict(X)

    print(str(accuracy_score(y, y_hat)))


    """
    PyTorch Test
    """

    from sklearn.preprocessing import LabelBinarizer
    l = LabelBinarizer()
    Y = torch.cuda.FloatTensor(l.fit_transform(y)).cuda()

    a = scipy_to_torch(X)

    start = time.time()
    m = PyTorchBayes(l)
    m.fit(a, Y)
    end = time.time() - start
    print(str(end))

    y_hat_gpu = m.predict(a)
    print(str(accuracy_score(y, y_hat_gpu.cpu().numpy())))

