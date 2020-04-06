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

import ctypes
import cudf
import cupy as cp
import numpy as np
import warnings

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, with_cupy_rmm
from cuml.utils.import_utils import has_cupy
from cuml.metrics import accuracy_score

cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

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
               int loss_type) except +

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
               int loss_type) except +

    void qnDecisionFunction(cumlHandle& cuml_handle,
                            float *X,
                            int N,
                            int D,
                            int C,
                            bool fit_intercept,
                            float *params,
                            bool X_col_major,
                            int loss_type,
                            float *scores) except +

    void qnDecisionFunction(cumlHandle& cuml_handle,
                            double *X,
                            int N,
                            int D,
                            int C,
                            bool fit_intercept,
                            double *params,
                            bool X_col_major,
                            int loss_type,
                            double *scores) except +

    void qnPredict(cumlHandle& cuml_handle,
                   float *X,
                   int N,
                   int D,
                   int C,
                   bool fit_intercept,
                   float *params,
                   bool X_col_major,
                   int loss_type,
                   float *preds) except +

    void qnPredict(cumlHandle& cuml_handle,
                   double *X,
                   int N,
                   int D,
                   int C,
                   bool fit_intercept,
                   double *params,
                   bool X_col_major,
                   int loss_type,
                   double *preds) except +


class QN(Base):
    """
    Quasi-Newton methods are used to either find zeroes or local maxima
    and minima of functions, and used by this class to optimize a cost
    function.

    Two algorithms are implemented underneath cuML's QN class, and which one
    is executed depends on the following rule:

    * Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
      regularization

    * Limited Memory BFGS (L-BFGS) otherwise.

    cuML's QN class can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant).

    Examples
    ---------
    .. code-block:: python

        import cudf
        import numpy as np

        # Both import methods supported
        # from cuml import QN
        from cuml.solvers import QN

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series( np.array([0.0, 0.0, 1.0, 1.0], dtype = np.float32) )

        solver = QN()
        solver.fit(X,y)

        # Note: for now, the coefficients also include the intercept in the
        # last position if fit_intercept=True
        print("Coefficients:")
        print(solver.coef_.copy_to_host())
        print("Intercept:")
        print(solver.intercept_.copy_to_host())

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([1,5], dtype = np.float32)
        X_new['col2'] = np.array([2,5], dtype = np.float32)

        preds = solver.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:
                    10.647417
                    0.3267412
                    -17.158297
        Intercept:
                    -17.158297
        Predictions:
                    0    0.0
                    1    1.0

    Parameters
    -----------
    loss: 'sigmoid', 'softmax', 'squared_loss' (default = 'squared_loss')
        'sigmoid' loss used for single class logistic regression
        'softmax' loss used for multiclass logistic regression
        'normal' used for normal/square loss
    fit_intercept: boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    l1_strength: float (default = 0.0)
        l1 regularization strength (if non-zero, will run OWL-QN, else L-BFGS).
        Note, that as in Scikit-learn, the bias will not be regularized.
    l2_strength: float (default = 0.0)
        l2 regularization strength. Note, that as in Scikit-learn, the bias
        will not be regularized.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    tol: float (default = 1e-3)
        The training process will stop if current_loss > previous_loss - tol
    linesearch_max_iter: int (default = 50)
        Max number of linesearch iterations per outer iteration of the
        algorithm.
    lbfgs_memory: int (default = 5)
        Rank of the lbfgs inverse-Hessian approximation. Method will use
        O(lbfgs_memory * D) memory.
    verbose: int (optional, default 0)
        Controls verbosity level of logging.

    Attributes
    -----------
    coef_ : array, shape (n_classes, n_features)
        The estimated coefficients for the linear regression model.
        Note: shape is (n_classes, n_features + 1) if fit_intercept = True.
    intercept_ : array (n_classes, 1)
        The independent term. If fit_intercept_ is False, will be 0.

    Notes
    ------
       This class contains implementations of two popular Quasi-Newton methods:

       - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal,
         Wright - Numerical Optimization (1999)]

       - Orthant-wise limited-memory quasi-newton (OWL-QN) [Andrew, Gao - ICML
         2007]
         <https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/>
    """

    def __init__(self, loss='sigmoid', fit_intercept=True,
                 l1_strength=0.0, l2_strength=0.0, max_iter=1000, tol=1e-3,
                 linesearch_max_iter=50, lbfgs_memory=5, verbose=0,
                 handle=None):

        super(QN, self).__init__(handle=handle, verbose=verbose)

        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.num_iter = 0
        self.coef_ = None

        if loss not in ['sigmoid', 'softmax', 'normal']:
            raise ValueError("loss " + str(loss) + " not supported.")

        self.loss = loss

    def _get_loss_int(self, loss):
        return {
            'sigmoid': 0,
            'softmax': 2,
            'normal': 1
        }[loss]

    @with_cupy_rmm
    def fit(self, X, y, convert_dtype=False):
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

        convert_dtype : bool, optional (default = False)
            When set to True, the fit method will, when necessary, convert
            y to be the same data type as X if they differ. This
            will increase memory used for the method.
        """

        cdef uintptr_t X_ptr, y_ptr
        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F',
                               check_dtype=[np.float32, np.float64])

        y_m, y_ptr, lab_rows, _, _ = \
            input_to_dev_array(y, check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_rows=n_rows, check_cols=1)

        self._num_classes = len(cp.unique(y_m))

        self.loss_type = self._get_loss_int(self.loss)
        if self.loss_type != 2 and self._num_classes > 2:
            raise ValueError("Only softmax (multinomial) loss supports more"
                             "than 2 classes.")

        if self.loss_type == 2 and self._num_classes <= 2:
            raise ValueError("Only softmax (multinomial) loss supports more"
                             "than 2 classes.")

        if self.loss_type == 0:
            self._num_classes_dim = self._num_classes - 1
        else:
            self._num_classes_dim = self._num_classes

        if self.fit_intercept:
            coef_size = (self.n_cols + 1, self._num_classes_dim)
        else:
            coef_size = (self.n_cols, self._num_classes_dim)

        self.coef_ = rmm.to_device(np.ones(coef_size, dtype=self.dtype))
        cdef uintptr_t coef_ptr = get_dev_array_ptr(self.coef_)

        cdef float objective32
        cdef double objective64
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef int num_iters

        if self.dtype == np.float32:
            qnFit(handle_[0],
                  <float*>X_ptr,
                  <float*>y_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <int> self._num_classes,
                  <bool> self.fit_intercept,
                  <float> self.l1_strength,
                  <float> self.l2_strength,
                  <int> self.max_iter,
                  <float> self.tol,
                  <int> self.linesearch_max_iter,
                  <int> self.lbfgs_memory,
                  <int> self.verbose,
                  <float*> coef_ptr,
                  <float*> &objective32,
                  <int*> &num_iters,
                  <bool> True,
                  <int> self.loss_type)

            self.objective = objective32

        else:
            qnFit(handle_[0],
                  <double*>X_ptr,
                  <double*>y_ptr,
                  <int>n_rows,
                  <int>self.n_cols,
                  <int> self._num_classes,
                  <bool> self.fit_intercept,
                  <double> self.l1_strength,
                  <double> self.l2_strength,
                  <int> self.max_iter,
                  <double> self.tol,
                  <int> self.linesearch_max_iter,
                  <int> self.lbfgs_memory,
                  <int> self.verbose,
                  <double*> coef_ptr,
                  <double*> &objective64,
                  <int*> &num_iters,
                  <bool> True,
                  <int> self.loss_type)

            self.objective = objective64

        self.num_iters = num_iters

        self.handle.sync()

        del X_m
        del y_m

        return self

    def _decision_function(self, X, convert_dtype=False):
        """
        Gives confidence score for X

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        Returns
        ----------
        y: array-like (device)
            Dense matrix (floats or doubles) of shape (n_samples, n_classes)
        """

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, self.dtype = \
            input_to_dev_array(X, check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_cols=self.n_cols)

        scores = rmm.to_device(zeros((self._num_classes_dim, n_rows),
                                     dtype=self.dtype, order='F'))

        cdef uintptr_t coef_ptr = get_dev_array_ptr(self.coef_)
        cdef uintptr_t scores_ptr = get_dev_array_ptr(scores)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            qnDecisionFunction(handle_[0],
                               <float*> X_ptr,
                               <int> n_rows,
                               <int> n_cols,
                               <int> self._num_classes,
                               <bool> self.fit_intercept,
                               <float*> coef_ptr,
                               <bool> True,
                               <int> self.loss_type,
                               <float*> scores_ptr)

        else:
            qnDecisionFunction(handle_[0],
                               <double*> X_ptr,
                               <int> n_rows,
                               <int> n_cols,
                               <int> self._num_classes,
                               <bool> self.fit_intercept,
                               <double*> coef_ptr,
                               <bool> True,
                               <int> self.loss_type,
                               <double*> scores_ptr)

        self.handle.sync()

        del X_m

        return scores

    def predict(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)
        """

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, self.dtype = \
            input_to_dev_array(X, check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_cols=self.n_cols)

        preds = rmm.to_device(zeros(n_rows, dtype=self.dtype))

        cdef uintptr_t coef_ptr = get_dev_array_ptr(self.coef_)
        cdef uintptr_t pred_ptr = get_dev_array_ptr(preds)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            qnPredict(handle_[0],
                      <float*> X_ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <int> self._num_classes,
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
                      <int> self._num_classes,
                      <bool> self.fit_intercept,
                      <double*> coef_ptr,
                      <bool> True,
                      <int> self.loss_type,
                      <double*> pred_ptr)

        self.handle.sync()

        del X_m

        return cudf.Series(preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def __getattr__(self, attr):
        if attr == 'intercept_':
            if self.fit_intercept:
                return self.coef_[-1]
            else:
                return rmm.to_device(zeros(1))
        else:
            raise AttributeError(attr)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable handle.
        if 'handle' in state:
            del state['handle']
        if 'coef_' in state:
            if state['coef_'] is not None:
                state['coef_'] = cudf.Series(state['coef_'].ravel())
        return state

    def __setstate__(self, state):
        super(QN, self).__init__(handle=None, verbose=state['verbose'])

        if 'coef_' in state and state['coef_'] is not None:
            if 'fit_intercept' in state and state['fit_intercept']:
                coef_size = (state['n_cols'] + 1, state['_num_classes'])
            else:
                coef_size = (state['n_cols'], state['_num_classes'])
            state['coef_'] = state['coef_'].to_gpu_array().reshape(coef_size)

        self.__dict__.update(state)

    def get_param_names(self):
        return ['loss', 'fit_intercept', 'l1_strength', 'l2_strength',
                'max_iter', 'tol', 'linesearch_max_iter', 'lbfgs_memory',
                'verbose']
