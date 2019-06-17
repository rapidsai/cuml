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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.solvers import QN
from cuml.common.base import Base

import numpy as np
import warnings

from cuml.utils import input_to_dev_array

supported_penalties = ['l1', 'l2', 'none']


class LogisticRegression(Base):
    """
    Logistic Goodness :)
    """

    def __init__(self, penalty='l2', tol=1e-3, C=1.0, fit_intercept=True,
                 class_weight=None, max_iter=1000, verbose=0, l1_ratio=None,
                 dual=None, handle=None):

        super(LogisticRegression, self).__init__(handle=handle, verbose=False)

        if dual:
            raise ValueError("`dual` parameter not supported.")

        if class_weight:
            raise ValueError("`class_weight` not supported.")

        if penalty not in supported_penalties or l1_ratio:
            raise ValueError("`penalty` " + str(penalty) + "not supported.")

        self.C = C
        self.penalty = penalty
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter=max_iter

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (floats or doubles) of shape (n_samples, 1).
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        y_m, _, _, _, _ = input_to_dev_array(y)

        try:
            import cupy as cp
            unique_labels = cp.unique(y_m)
        except ImportError:
            warnings.warn("Using NumPy for number of class detection,"
                          "install CuPy for faster processing.")
            unique_labels = np.unique(y_m.copy_to_host())

        num_classes = len(unique_labels)

        if len(unique_labels) > 2:
            loss = 'softmax'
        else:
            loss = 'sigmoid'

        if self.penalty == 'l1':
            l1_ratio = 1.0 / self.C
            l2_ratio = 0.0
        else:
            l1_ratio = 0.0
            l2_ratio = 1.0 / self.C

        self.qn = QN(loss=loss, fit_intercept=self.fit_intercept,
                     l1_ratio=l1_ratio, l2_ratio=l2_ratio,
                     max_iter=self.max_iter, tol=self.tol,
                     verbose=self.verbose, num_classes=num_classes,
                     handle=self.handle)

        self.qn.fit(X, y_m)
        self.coef_ = self.qn.coef_

        return self

    def predict(self, X):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """
        return self.qn.predict(X)
