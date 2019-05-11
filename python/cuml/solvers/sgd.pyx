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

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base

cdef extern from "solver/solver_c.h" namespace "ML::Solver":

    cdef void sgdFit(float *input,
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
                     int n_iter_no_change)

    cdef void sgdFit(double *input,
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
                     int n_iter_no_change)

    cdef void sgdPredict(const float *input,
                         int n_rows,
                         int n_cols,
                         const float *coef,
                         float intercept,
                         float *preds,
                         int loss)

    cdef void sgdPredict(const double *input,
                         int n_rows,
                         int n_cols,
                         const double *coef,
                         double intercept,
                         double *preds,
                         int loss)

    cdef void sgdPredictBinaryClass(const float *input,
                                    int n_rows,
                                    int n_cols,
                                    const float *coef,
                                    float intercept,
                                    float *preds,
                                    int loss)

    cdef void sgdPredictBinaryClass(const double *input,
                                    int n_rows,
                                    int n_cols,
                                    const double *coef,
                                    double intercept,
                                    double *preds,
                                    int loss)


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
        pred_data['col1'] = np.asarray([3, 2], dtype=datatype)
        pred_data['col2'] = np.asarray([5, 5], dtype=datatype)

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
                 batch_size=32, n_iter_no_change=5):

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
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        cdef uintptr_t X_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix must be a cuDF dataframe or Numpy ndarray"
            raise TypeError(msg)

        X_ptr = self._get_dev_array_ptr(X_m)

        cdef uintptr_t y_ptr
        if (isinstance(y, cudf.Series)):
            y_ptr = self._get_cudf_column_ptr(y)
        elif (isinstance(y, np.ndarray)):
            y_m = cuda.to_device(y)
            y_ptr = self._get_dev_array_ptr(y_m)
        else:
            msg = "y vector must be a cuDF series or Numpy ndarray"
            raise TypeError(msg)

        self.n_alpha = 1

        self.coef_ = cudf.Series(np.zeros(self.n_cols,
                                          dtype=self.gdf_datatype))
        cdef uintptr_t coef_ptr = self._get_cudf_column_ptr(self.coef_)

        cdef float c_intercept1
        cdef double c_intercept2

        if self.gdf_datatype.type == np.float32:
            sgdFit(<float*>X_ptr,
                   <int>self.n_rows,
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
            sgdFit(<double*>X_ptr,
                   <int>self.n_rows,
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

        return self

    def predict(self, X):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        cdef uintptr_t X_ptr
        if (isinstance(X, cudf.DataFrame)):
            pred_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            n_rows = len(X)
            n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            pred_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_rows = X.shape[0]
            n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        X_ptr = self._get_dev_array_ptr(X_m)

        cdef uintptr_t coef_ptr = self._get_cudf_column_ptr(self.coef_)
        preds = cudf.Series(np.zeros(n_rows, dtype=pred_datatype))
        cdef uintptr_t preds_ptr = self._get_cudf_column_ptr(preds)

        if pred_datatype.type == np.float32:
            sgdPredict(<float*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <float*>coef_ptr,
                       <float>self.intercept_,
                       <float*>preds_ptr,
                       <int>self.loss)
        else:
            sgdPredict(<double*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <double*>coef_ptr,
                       <double>self.intercept_,
                       <double*>preds_ptr,
                       <int>self.loss)

        del(X_m)

        return preds

    def predictClass(self, X):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        ----------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        cdef uintptr_t X_ptr
        if (isinstance(X, cudf.DataFrame)):
            pred_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            n_rows = len(X)
            n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            pred_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_rows = X.shape[0]
            n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        X_ptr = self._get_ctype_ptr(X_m)

        cdef uintptr_t coef_ptr = self._get_cudf_column_ptr(self.coef_)
        preds = cudf.Series(np.zeros(n_rows, dtype=pred_datatype))
        cdef uintptr_t preds_ptr = self._get_cudf_column_ptr(preds)

        if pred_datatype.type == np.float32:
            sgdPredictBinaryClass(<float*>X_ptr,
                                  <int>n_rows,
                                  <int>n_cols,
                                  <float*>coef_ptr,
                                  <float>self.intercept_,
                                  <float*>preds_ptr,
                                  <int>self.loss)
        else:
            sgdPredictBinaryClass(<double*>X_ptr,
                                  <int>n_rows,
                                  <int>n_cols,
                                  <double*>coef_ptr,
                                  <double>self.intercept_,
                                  <double*>preds_ptr,
                                  <int>self.loss)

        del(X_m)

        return preds
