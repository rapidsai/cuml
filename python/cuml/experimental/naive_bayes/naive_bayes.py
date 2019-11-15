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
import torch

import cupy as cp

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from cuml.preprocessing import LabelBinarizer


class MultinomialNB(object):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.n_features_ = None

    def fit(self, X, y, _partial=False, _classes=None):

        print(str(X.shape))
        print(str(y.shape))

        label_binarizer = LabelBinarizer()

        y1 = cp.fromDlpack(to_dlpack(y))

        print("Y=" + str(y.dtype))
        print("y=" + str(y1))

        res = label_binarizer.fit_transform(y1)

        Y = from_dlpack(res.toDlpack())

        self.n_features_ = X.shape[1]
        self._init_counters(Y.shape[1], self.n_features_)

        self.classes_ = label_binarizer.classes_
        self.n_classes_ = len(label_binarizer.classes_)

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
        pass

    def predict(self, X):
        jll = self._joint_log_likelihood(X)

        _, indices = torch.max(jll, 1)
        return indices

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = torch.zeros(n_effective_classes).cuda()
        self.feature_count_ = torch.zeros(n_effective_classes, n_features).cuda()

    def _count(self, X, Y):
        self.feature_count_ += torch.sparse.mm(X.t(), Y.float()).t()
        self.class_count_ += Y.sum(axis=0)

    def _update_class_log_prior(self, class_prior=None):

        if class_prior is not None:

            if len(class_prior != self.n_classes):
                raise ValueError("Number of classes must match number of priors")

            self.class_log_prior_ = torch.log(class_prior)

        elif self.fit_prior:
            log_class_count = torch.log(self.class_count_)
            self.class_log_prior_ = log_class_count - \
                                    torch.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = torch.full(self.n_classes_,
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
