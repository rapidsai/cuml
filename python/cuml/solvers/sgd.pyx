# Copyright (c) 2018, NVIDIA CORPORATION.
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
import cupy as cp

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.common.handle cimport cumlHandle
from cuml.common import input_to_cuml_array, with_cupy_rmm

cdef extern from "cuml/solvers/solver.hpp" namespace "ML::Solver":

    cdef void sgdFit(cumlHandle& handle,
                     float *input,
                     int n_rows,
                     int n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     int batch_size,
                     int epochs,
                     int lr_type,
                     float eta0,
                     float power_t,
                     int loss,
                     int penalty,
                     float alpha,
                     float l1_ratio,
                     bool shuffle,
                     float tol,
                     int n_iter_no_change) except +

    cdef void sgdFit(cumlHandle& handle,
                     double *input,
                     int n_rows,
                     int n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     int batch_size,
                     int epochs,
                     int lr_type,
                     double eta0,
                     double power_t,
                     int loss,
                     int penalty,
                     double alpha,
                     double l1_ratio,
                     bool shuffle,
                     double tol,
                     int n_iter_no_change) except +

    cdef void sgdPredict(cumlHandle& handle,
                         const float *input,
                         int n_rows,
                         int n_cols,
                         const float *coef,
                         float intercept,
                         float *preds,
                         int loss) except +

    cdef void sgdPredict(cumlHandle& handle,
                         const double *input,
                         int n_rows,
                         int n_cols,
                         const double *coef,
                         double intercept,
                         double *preds,
                         int loss) except +

    cdef void sgdPredictBinaryClass(cumlHandle& handle,
                                    const float *input,
                                    int n_rows,
                                    int n_cols,
                                    const float *coef,
                                    float intercept,
                                    float *preds,
                                    int loss) except +

    cdef void sgdPredictBinaryClass(cumlHandle& handle,
                                    const double *input,
                                    int n_rows,
                                    int n_cols,
                                    const double *coef,
                                    double intercept,
                                    double *preds,
                                    int loss) except +


class SGD(Base):
    """
    Stochastic Gradient Descent is a very common machine learning algorithm
    where one optimizes some cost function via gradient steps. This makes SGD
    very attractive for large problems when the exact solution is hard or even
    impossible to find.

    cuML's SGD algorithm accepts a numpy matrix or a cuDF DataFrame as the
    input dataset. The SGD algorithm currently works with linear regression,
    ridge regression and SVM models.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.solvers import SGD as cumlSGD
        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series(np.array([1, 1, 2, 2], dtype=np.float32))
        pred_data = cudf.DataFrame()
        pred_data['col1'] = np.asarray([3, 2], dtype=dtype)
        pred_data['col2'] = np.asarray([5, 5], dtype=dtype)
        cu_sgd = cumlSGD(learning_rate=lrate, eta0=0.005, epochs=2000,
                        fit_intercept=True, batch_size=2,
                        tol=0.0, penalty=penalty, loss=loss)
        cu_sgd.fit(X, y)
        cu_pred = cu_sgd.predict(pred_data).to_array()
        print(" cuML intercept : ", cu_sgd.intercept_)
        print(" cuML coef : ", cu_sgd.coef_)
        print("cuML predictions : ", cu_pred)

    Output:

    .. code-block:: python

        cuML intercept :  0.004561662673950195
        cuML coef :  0      0.9834546
                    1    0.010128272
                   dtype: float32
        cuML predictions :  [3.0055666 2.0221121]


    Parameters
    -----------
    loss : 'hinge', 'log', 'squared_loss' (default = 'squared_loss')
        'hinge' uses linear SVM
        'log' uses logistic regression
        'squared_loss' uses linear regression
    penalty: 'none', 'l1', 'l2', 'elasticnet' (default = 'none')
        'none' does not perform any regularization
        'l1' performs L1 norm (Lasso) which minimizes the sum of the abs value
        of coefficients
        'l2' performs L2 norm (Ridge) which minimizes the sum of the square of
        the coefficients
        'elasticnet' performs Elastic Net regularization which is a weighted
        average of L1 and L2 norms
    alpha: float (default = 0.0001)
        The constant value which decides the degree of regularization
    fit_intercept : boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    epochs : int (default = 1000)
        The number of times the model should iterate through the entire dataset
        during training (default = 1000)
    tol : float (default = 1e-3)
        The training process will stop if current_loss > previous_loss - tol
    shuffle : boolean (default = True)
        True, shuffles the training data after each epoch
        False, does not shuffle the training data after each epoch
    eta0 : float (default = 0.001)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    learning_rate : 'optimal', 'constant', 'invscaling', \
                    'adaptive' (default = 'constant')
        optimal option supported in the next version
        constant keeps the learning rate constant
        adaptive changes the learning rate if the training loss or the
        validation accuracy does not improve for n_iter_no_change epochs.
        The old learning rate is generally divide by 5
    n_iter_no_change : int (default = 5)
        the number of epochs to train without any imporvement in the model
    output_type : {'input', 'cudf', 'cupy', 'numpy'}, optional
        Variable to control output type of the results and attributes of
        the estimators. If None, it'll inherit the output type set at the
        module level, cuml.output_type. If set, the estimator will override
        the global option for its behavior.

    """

    def __init__(self, loss='squared_loss', penalty='none', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, epochs=1000, tol=1e-3,
                 shuffle=True, learning_rate='constant', eta0=0.001,
                 power_t=0.5, batch_size=32, n_iter_no_change=5, handle=None,
                 output_type=None):

        if loss in ['hinge', 'log', 'squared_loss']:
            self.loss = self._get_loss_int(loss)
        else:
            msg = "loss {!r} is not supported"
            raise TypeError(msg.format(loss))

        if penalty in ['none', 'l1', 'l2', 'elasticnet']:
            self.penalty = self._get_penalty_int(penalty)
        else:
            msg = "penalty {!r} is not supported"
            raise TypeError(msg.format(penalty))

        super(SGD, self).__init__(handle=handle, verbose=False,
                                  output_type=output_type)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.tol = tol
        self.shuffle = shuffle
        self.eta0 = eta0
        self.power_t = power_t

        if learning_rate in ['optimal', 'constant', 'invscaling', 'adaptive']:
            self.learning_rate = learning_rate

            if learning_rate in ["constant", "invscaling", "adaptive"]:
                if self.eta0 <= 0.0:
                    raise ValueError("eta0 must be > 0")

            if learning_rate == 'optimal':
                self.lr_type = 0

                raise TypeError("This option will be supported in the future")

                if self.alpha == 0:
                    raise ValueError("alpha must be > 0 since "
                                     "learning_rate is 'optimal'. alpha is "
                                     "used to compute the optimal learning "
                                     " rate.")

            elif learning_rate == 'constant':
                self.lr_type = 1
                self.lr = eta0
            elif learning_rate == 'invscaling':
                self.lr_type = 2
            elif learning_rate == 'adaptive':
                self.lr_type = 3
        else:
            msg = "learning rate {!r} is not supported"
            raise TypeError(msg.format(learning_rate))

        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.intercept_value = 0.0
        self._coef_ = None  # accessed via coef_
        self.intercept_ = None

    def _check_alpha(self, alpha):
        for el in alpha:
            if el <= 0.0:
                msg = "alpha values have to be positive"
                raise TypeError(msg.format(alpha))

    def _get_loss_int(self, loss):
        return {
            'squared_loss': 0,
            'log': 1,
            'hinge': 2,
        }[loss]

    def _get_penalty_int(self, penalty):
        return {
            'none': 0,
            'l1': 1,
            'l2': 2,
            'elasticnet': 3
        }[penalty]

    @generate_docstring()
    @with_cupy_rmm
    def fit(self, X, y, convert_dtype=False):
        """
        Fit the model with X and y.

        """
        self._set_output_type(X)
        self._set_target_dtype(y)

        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])

        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)

        _estimator_type = getattr(self, '_estimator_type', None)
        if _estimator_type == "classifier":
            self._classes_ = CumlArray(cp.unique(y_m))

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t y_ptr = y_m.ptr

        self.n_alpha = 1

        self._coef_ = CumlArray.zeros(self.n_cols,
                                      dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float c_intercept1
        cdef double c_intercept2
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            sgdFit(handle_[0],
                   <float*>X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <float*>y_ptr,
                   <float*>coef_ptr,
                   <float*>&c_intercept1,
                   <bool>self.fit_intercept,
                   <int>self.batch_size,
                   <int>self.epochs,
                   <int>self.lr_type,
                   <float>self.eta0,
                   <float>self.power_t,
                   <int>self.loss,
                   <int>self.penalty,
                   <float>self.alpha,
                   <float>self.l1_ratio,
                   <bool>self.shuffle,
                   <float>self.tol,
                   <int>self.n_iter_no_change)

            self.intercept_ = c_intercept1
        else:
            sgdFit(handle_[0],
                   <double*>X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <double*>y_ptr,
                   <double*>coef_ptr,
                   <double*>&c_intercept2,
                   <bool>self.fit_intercept,
                   <int>self.batch_size,
                   <int>self.epochs,
                   <int>self.lr_type,
                   <double>self.eta0,
                   <double>self.power_t,
                   <int>self.loss,
                   <int>self.penalty,
                   <double>self.alpha,
                   <double>self.l1_ratio,
                   <bool>self.shuffle,
                   <double>self.tol,
                   <int>self.n_iter_no_change)

            self.intercept_ = c_intercept2

        self.handle.sync()

        del X_m
        del y_m

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        """
        output_type = self._get_output_type(X)

        X_m, n_rows, n_cols, self.dtype = \
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
            sgdPredict(handle_[0],
                       <float*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <float*>coef_ptr,
                       <float>self.intercept_,
                       <float*>preds_ptr,
                       <int>self.loss)
        else:
            sgdPredict(handle_[0],
                       <double*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <double*>coef_ptr,
                       <double>self.intercept_,
                       <double*>preds_ptr,
                       <int>self.loss)

        self.handle.sync()

        del(X_m)

        return preds.to_output(output_type)

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predictClass(self, X, convert_dtype=False):
        """
        Predicts the y for X.

        """
        output_type = self._get_output_type(X)
        out_dtype = self._get_target_dtype()

        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t coef_ptr = self._coef_.ptr
        preds = CumlArray.zeros(n_rows, dtype=dtype)
        cdef uintptr_t preds_ptr = preds.ptr
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            sgdPredictBinaryClass(handle_[0],
                                  <float*>X_ptr,
                                  <int>n_rows,
                                  <int>n_cols,
                                  <float*>coef_ptr,
                                  <float>self.intercept_,
                                  <float*>preds_ptr,
                                  <int>self.loss)
        else:
            sgdPredictBinaryClass(handle_[0],
                                  <double*>X_ptr,
                                  <int>n_rows,
                                  <int>n_cols,
                                  <double*>coef_ptr,
                                  <double>self.intercept_,
                                  <double*>preds_ptr,
                                  <int>self.loss)

        self.handle.sync()

        del(X_m)

        return preds.to_output(output_type=output_type, output_dtype=out_dtype)
