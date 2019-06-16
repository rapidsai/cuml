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

import ctypes
import cudf
import numpy as np
import cupy

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros


cdef extern from "glm/glm.hpp" namespace "ML::GLM":

    void qnFit(cumlHandle& cuml_handle,
               float *X,
               float *y,
               int N,
               int D,
               int C,
               bool fit_intercept,
               float l1,
               float l2,
               int max_iter,
               float grad_tol,
               int linesearch_max_iter,
               int lbfgs_memory,
               int verbosity,
               float *w0,
               float *f,
               int *num_iters,
               bool X_col_major,
               int loss_type)

    void qnFit(cumlHandle& cuml_handle,
               double *X,
               double *y,
               int N,
               int D,
               int C,
               bool fit_intercept,
               double l1,
               double l2,
               int max_iter,
               double grad_tol,
               int linesearch_max_iter,
               int lbfgs_memory,
               int verbosity,
               double *w0,
               double *f,
               int *num_iters,
               bool X_col_major,
               int loss_type)

    void qnPredict(cumlHandle& cuml_handle,
                   float *X,
                   int N,
                   int D,
                   int C,
                   bool fit_intercept,
                   float *params,
                   bool X_col_major,
                   int loss_type,
                   float *preds)

    void qnPredict(cumlHandle& cuml_handle,
                   double *X,
                   int N,
                   int D,
                   int C,
                   bool fit_intercept,
                   double *params,
                   bool X_col_major,
                   int loss_type,
                   double *preds)


class QN(Base):
    """
    Quasi-Newton Goodness :)
    """

    def __init__(self, loss='sigmoid', fit_intercept=True,
                 l1_ratio=0.15, max_iter=1000, tol=1e-3,
                 linesearch_max_iter=1000, lbfgs_memory=5, verbose=False,
                 num_classes=1):

        super(CD, self).__init__(handle=handle, verbose=verbose)

        self.fit_intercept = fit_intercept
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.alpha = 0.0
        self.num_iter = 0
        self.loss_type = self._get_loss_int(loss)

    def _get_loss_int(self, loss):
        return {
            'sigmoid': 0,
            'softmax': 1,
            'normal': 2
        }[loss]

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

        cdef uintptr_t X_ptr, y_ptr
        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X)

        y_m, y_ptr, _, _, _ = \
            input_to_dev_array(y)

        self.num_classes = len(cupy.unique(y_m))

        if self.loss_type != 1 and self.num_classes > 2:
            raise ValueError("Only softmax (multinomial) loss supports more"
                             "than 2 classes.")

        self.coef_ = cuda.to_device(zeros(n_cols, self.num_classes,
                                          dtype=self.dtype))

        cdef uintptr_t coef_ptr = get_dev_array_ptr(self.coef_)

        cdef float intercept32, objective32
        cdef double intercept64, objective64
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            qnFit(handle_[0],
                  <float*>X_ptr,
                  <float*>y_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <int> self.num_classes,
                  <bool> self.fit_intercept,
                  <float> self.l1_ratio,
                  <float> 1.0-self.l2_ratio,
                  <int> self.max_iter,
                  <float> self.tol,
                  <int> self.linesearch_max_iter,
                  <int> self.lbfgs_memory,
                  <int> 0,
                  <float*> coef_ptr,
                  <float*> objective32,
                  <int*> self.num_iters,
                  <bool> True,
                  <int> self.loss_type)

        else:
            qnFit(handle_[0],
                  <double*>X_ptr,
                  <double*>y_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <int> self.num_classes,
                  <bool> self.fit_intercept,
                  <double> self.l1,
                  <double> self.l2,
                  <int> self.max_iter,
                  <double> self.tol,
                  <int> self.linesearch_max_iter,
                  <int> self.lbfgs_memory,
                  <int> 0,
                  <double*> coef_ptr,
                  <double*> objective64,
                  <int*> self.num_iters,
                  <bool> True,
                  <int> self.loss_type)

        self.handle.sync()

        del X_m
        del y_m

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

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, self.dtype = \
            input_to_dev_array(X, check_dtype=self.dtype,
                               check_cols=self.n_cols)

        preds = cuda.to_device(zeros(n_rows, self.num_classes,
                                     dtype=self.dtype))

        cdef uintptr_t coef_ptr = get_dev_array_ptr(self.coef_)
        cdef uintptr_t pred_ptr = get_dev_array_ptr(preds)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            qnPredict(handle_[0],
                      <float*> X_ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <int> self.num_classes,
                      <bool> self.fit_intercept,
                      <float*> coef_ptr,
                      <bool> True,
                      <int> self.loss_type,
                      <float*> pred_ptr)

        else:
            qnPredict(handle_[0],
                      <double*> X_ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <int> self.num_classes,
                      <bool> self.fit_intercept,
                      <double*> coef_ptr,
                      <bool> True,
                      <int> self.loss_type,
                      <double*> pred_ptr)

        self.handle.sync()

        del X_m

        return cudf.Series(preds)
