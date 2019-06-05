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


import ctypes
import cudf
import numpy as np

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

cdef extern from "solver/solver.hpp" namespace "ML::Solver":

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
    ---------
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
    eta0 : float (default = 0.0)
        Initial learning rate
    power_t : float (default = 0.5)
        The exponent used for calculating the invscaling learning rate
    learning_rate : 'optimal', 'constant', 'invscaling',
                    'adaptive' (default = 'constant')
        optimal option supported in the next version
        constant keeps the learning rate constant
        adaptive changes the learning rate if the training loss or the
        validation accuracy does not improve for n_iter_no_change epochs.
        The old learning rate is generally divide by 5
    n_iter_no_change : int (default = 5)
        the number of epochs to train without any imporvement in the model
    Notes
    ------
    For additional docs, see `scikitlearn's OLS
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>
    """

    def __init__(self, loss='squared_loss', penalty='none', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, epochs=1000, tol=1e-3,
                 shuffle=True, learning_rate='constant', eta0=0.0, power_t=0.5,
                 batch_size=32, n_iter_no_change=5, handle=None):

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

        super(SGD, self).__init__(handle=handle, verbose=False)
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
        self.coef_ = None
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

        self.n_alpha = 1

        self.coef_ = cudf.Series(zeros(self.n_cols,
                                          dtype=self.dtype))
        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)

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
        X_m, X_ptr, n_rows, n_cols, self.dtype = input_to_dev_array(X)

        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)
        preds = cudf.Series(zeros(n_rows, dtype=self.dtype))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)

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

        return preds

    def predictClass(self, X):
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
        X_m, X_ptr, n_rows, n_cols, dtype = input_to_dev_array(X)

        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)
        preds = cudf.Series(zeros(n_rows, dtype=dtype))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)
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

        return preds
