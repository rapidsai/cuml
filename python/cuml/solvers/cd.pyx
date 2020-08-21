# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

import ctypes
import cudf
import numpy as np

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from cuml.common.handle cimport cumlHandle
from cuml.common import get_cudf_column_ptr
from cuml.common import get_dev_array_ptr
from cuml.common import input_to_dev_array
from cuml.common import zeros
from cuml.common.input_utils import input_to_cuml_array


cdef extern from "cuml/solvers/solver.hpp" namespace "ML::Solver":

    cdef void cdFit(cumlHandle& handle,
                    float *input,
                    int n_rows,
                    int n_cols,
                    float *labels,
                    float *coef,
                    float *intercept,
                    bool fit_intercept,
                    bool normalize,
                    int epochs,
                    int loss,
                    float alpha,
                    float l1_ratio,
                    bool shuffle,
                    float tol) except +

    cdef void cdFit(cumlHandle& handle,
                    double *input,
                    int n_rows,
                    int n_cols,
                    double *labels,
                    double *coef,
                    double *intercept,
                    bool fit_intercept,
                    bool normalize,
                    int epochs,
                    int loss,
                    double alpha,
                    double l1_ratio,
                    bool shuffle,
                    double tol) except +

    cdef void cdPredict(cumlHandle& handle,
                        const float *input,
                        int n_rows,
                        int n_cols,
                        const float *coef,
                        float intercept,
                        float *preds,
                        int loss) except +

    cdef void cdPredict(cumlHandle& handle,
                        const double *input,
                        int n_rows,
                        int n_cols,
                        const double *coef,
                        double intercept,
                        double *preds,
                        int loss) except +


class CD(Base):
    """
    Coordinate Descent (CD) is a very common optimization algorithm that
    minimizes along coordinate directions to find the minimum of a function.

    cuML's CD algorithm accepts a numpy matrix or a cuDF DataFrame as the
    input dataset.algorithm The CD algorithm currently works with linear
    regression and ridge, lasso, and elastic-net penalties.

    Examples
    ---------
    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.solvers import CD as cumlCD

        cd = cumlCD(alpha=0.0)

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        reg = cd.fit(X,y)

        print("Coefficients:")
        print(reg.coef_)
        print("intercept:")
        print(reg.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)

        preds = cd.predict(X_new)

        print(preds)

    Output:

    .. code-block:: python

        Coefficients:
                    0 1.0019531
                    1 1.9980469
        Intercept:
                    3.0
        Preds:
                    0 15.997
                    1 14.995

    Parameters
    -----------
    loss : 'squared_loss' (Only 'squared_loss' is supported right now)
       'squared_loss' uses linear regression
    alpha: float (default = 0.0001)
        The constant value which decides the degree of regularization.
        'alpha = 0' is equivalent to an ordinary least square, solved by the
        LinearRegression object.
    l1_ratio: float (default = 0.15)
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For
        l1_ratio = 0 the penalty is an L2 penalty.
        For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1,
        the penalty is a combination of L1 and L2.
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.
    max_iter : int (default = 1000)
        The number of times the model should iterate through the entire
        dataset during training (default = 1000)
    tol : float (default = 1e-3)
       The tolerance for the optimization: if the updates are smaller than tol,
       solver stops.
    shuffle : boolean (default = True)
       If set to ‘True’, a random coefficient is updated every iteration rather
       than looping over features sequentially by default.
       This (setting to ‘True’) often leads to significantly faster convergence
       especially when tol is higher than 1e-4.

    """

    def __init__(self, loss='squared_loss', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, normalize=False, max_iter=1000, tol=1e-3,
                 shuffle=True, handle=None, output_type=None):

        if loss in ['squared_loss']:
            self.loss = self._get_loss_int(loss)
        else:
            msg = "loss {!r} is not supported"
            raise NotImplementedError(msg.format(loss))

        super(CD, self).__init__(handle=handle, verbose=False,
                                 output_type=output_type)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.intercept_value = 0.0
        self._coef_ = None   # accessed via estimator.coef_
        self.intercept_ = None

    def _check_alpha(self, alpha):
        for el in alpha:
            if el <= 0.0:
                msg = "alpha values have to be positive"
                raise TypeError(msg.format(alpha))

    def _get_loss_int(self, loss):
        return {
            'squared_loss': 0,
        }[loss]

    @generate_docstring()
    def fit(self, X, y, convert_dtype=False):
        """
        Fit the model with X and y.

        """

        self._set_output_type(X)

        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])

        y_m, *_ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t y_ptr = y_m.ptr

        self.n_alpha = 1

        self._coef_ = CumlArray.zeros(self.n_cols, dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float c_intercept1
        cdef double c_intercept2
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            cdFit(handle_[0],
                  <float*>X_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <float*>y_ptr,
                  <float*>coef_ptr,
                  <float*>&c_intercept1,
                  <bool>self.fit_intercept,
                  <bool>self.normalize,
                  <int>self.max_iter,
                  <int>self.loss,
                  <float>self.alpha,
                  <float>self.l1_ratio,
                  <bool>self.shuffle,
                  <float>self.tol)

            self.intercept_ = c_intercept1
        else:
            cdFit(handle_[0],
                  <double*>X_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <double*>y_ptr,
                  <double*>coef_ptr,
                  <double*>&c_intercept2,
                  <bool>self.fit_intercept,
                  <bool>self.normalize,
                  <int>self.max_iter,
                  <int>self.loss,
                  <double>self.alpha,
                  <double>self.l1_ratio,
                  <bool>self.shuffle,
                  <double>self.tol)

            self.intercept_ = c_intercept2

        self.handle.sync()

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        """
        out_type = self._get_output_type(X)

        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t coef_ptr = self._coef_.ptr

        preds = CumlArray.zeros(n_rows, dtype=self.dtype)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            cdPredict(handle_[0],
                      <float*>X_ptr,
                      <int>n_rows,
                      <int>n_cols,
                      <float*>coef_ptr,
                      <float>self.intercept_,
                      <float*>preds_ptr,
                      <int>self.loss)
        else:
            cdPredict(handle_[0],
                      <double*>X_ptr,
                      <int>n_rows,
                      <int>n_cols,
                      <double*>coef_ptr,
                      <double>self.intercept_,
                      <double*>preds_ptr,
                      <int>self.loss)

        self.handle.sync()

        del(X_m)

        return preds.to_output(out_type)
