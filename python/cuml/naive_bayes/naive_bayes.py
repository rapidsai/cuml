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

import math

import cupy as cp

from cuml.preprocessing import LabelBinarizer


class MultinomialNB(object):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):

        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_ = None
        self.n_classes_ = 0

        self.n_features_ = None

    def fit(self, X, y, classes=None, _sparse_labels=False):

        if not _sparse_labels:
            label_binarizer = LabelBinarizer(sparse_output=False)
            Y = label_binarizer.fit_transform(y)
            self.classes_ = label_binarizer.classes_
            self.n_classes_ = label_binarizer.classes_.shape[0]
        else:
            Y = y
            self.classes_ = classes
            self.n_classes_ = classes.shape[0]

        self.n_features_ = X.shape[1]
        self._init_counters(Y.shape[1], self.n_features_)

        self._count(X, Y)
        self._update_feature_log_prob(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incremental fit on a batch of samples

        :param X:
        :param y:
        :param classes:
        :param sample_weight:
        :return:
        """
        return self.fit(X, y, _classes=classes)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        indices = cp.argmax(jll, axis=1)
        return indices

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = cp.zeros(n_effective_classes, dtype=cp.float32)
        self.feature_count_ = cp.zeros((n_effective_classes, n_features),
                                       dtype=cp.float32)

    def _count(self, X, Y):

        feature_count_ = X.T.dot(Y).T

        self.feature_count_ += feature_count_
        self.class_count_ += Y.sum(axis=0).reshape(self.n_classes_)

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if class_prior.shape[0] != self.n_classes:
                raise ValueError("Number of classes must match number of priors")

            self.class_log_prior_ = cp.log(class_prior)

        elif self.fit_prior:
            log_class_count = cp.log(self.class_count_)
            self.class_log_prior_ = log_class_count - \
                                    cp.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = cp.full(self.n_classes_,
                                            -math.log(self.n_classes_))

    def _update_feature_log_prob(self, alpha):

        """ apply add-lambda smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = (cp.log(smoothed_fc) - cp.log(smoothed_cc))

    def _joint_log_likelihood(self, X):
        """ Calculate the posterior log probability of the samples X """
        ret = X.dot(self.feature_log_prob_.T)
        ret += self.class_log_prior_.T
        return ret
