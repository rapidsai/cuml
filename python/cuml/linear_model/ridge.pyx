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

import ctypes
import cudf
import numpy as np
from collections import defaultdict
from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.metrics.base import RegressorMixin
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

cdef extern from "glm/glm.hpp" namespace "ML::GLM":

    cdef void ridgeFit(cumlHandle& handle,
                       float *input,
                       int n_rows,
                       int n_cols,
                       float *labels,
                       float *alpha,
                       int n_alpha,
                       float *coef,
                       float *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo)

    cdef void ridgeFit(cumlHandle& handle,
                       double *input,
                       int n_rows,
                       int n_cols,
                       double *labels,
                       double *alpha,
                       int n_alpha,
                       double *coef,
                       double *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo)

    cdef void ridgePredict(cumlHandle& handle,
                           const float *input,
                           int n_rows,
                           int n_cols,
                           const float *coef,
                           float intercept,
                           float *preds)

    cdef void ridgePredict(cumlHandle& handle,
                           const double *input,
                           int n_rows,
                           int n_cols,
                           const double *coef,
                           double intercept,
                           double *preds)


class Ridge(Base, RegressorMixin):

    """
    Ridge extends LinearRegression by providing L2 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can reduce the variance of the predictors, and improves
    the conditioning of the problem.

    cuML's Ridge can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant),
    as well as cuDF DataFrames. It provides 3 algorithms: SVD, Eig and CD to
    fit a linear model. SVD is more stable, but Eig (default) is much faster.
    CD uses Coordinate Descent and can be faster when data is large.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf

        # Both import methods supported
        from cuml import Ridge
        from cuml.linear_model import Ridge

        alpha = np.array([1e-5])
        ridge = Ridge(alpha = alpha, fit_intercept = True, normalize = False,
                      solver = "eig")

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        result_ridge = ridge.fit(X, y)
        print("Coefficients:")
        print(result_ridge.coef_)
        print("Intercept:")
        print(result_ridge.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_ridge.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 1.0000001
                    1 1.9999998

        Intercept:
                    3.0

        Preds:

                    0 15.999999
                    1 14.999999

    Parameters
    -----------
    alpha : float or double
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization. Array input will be supported later.
    solver : 'eig' or 'svd' or 'cd' (default = 'eig')
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
        CD or Coordinate Descent is very fast and is suitable for large
        problems.
    fit_intercept : boolean (default = True)
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2
        norm.
        If False, no scaling will be done.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If fit_intercept_ is False, will be 0.

    Notes
    ------
    Ridge provides L2 regularization. This means that the coefficients can
    shrink to become very small, but not zero. This can cause issues of
    interpretabiliy on the coefficients.
    Consider using Lasso, or thresholding small coefficients to zero.

    **Applications of Ridge**

        Ridge Regression is used in the same way as LinearRegression, but is
        used frequently as it does not suffer from multicollinearity issues.
        Ridge is used in insurance premium prediction, stock market analysis
        and much more.


    For additional docs, see `scikitlearn's Ridge
    <https://github.com/rapidsai/notebooks/blob/master/cuml/ridge_regression_demo.ipynb>`_.
    """

    def __init__(self, alpha=1.0, solver='eig', fit_intercept=True,
                 normalize=False, handle=None):

        """
        Initializes the linear ridge regression class.

        Parameters
        ----------
        solver : Type: string. 'eig' (default) and 'svd' are supported
        algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self._check_alpha(alpha)
        super(Ridge, self).__init__(handle=handle, verbose=False)
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        if solver in ['svd', 'eig', 'cd']:
            self.solver = solver
            self.algo = self._get_algorithm_int(solver)
        else:
            msg = "solver {!r} is not supported"
            raise TypeError(msg.format(solver))
        self.intercept_value = 0.0

    def _check_alpha(self, alpha):
        if alpha <= 0.0:
            msg = "alpha value has to be positive"
            raise TypeError(msg.format(alpha))

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1,
            'cd': 2
        }[algorithm]

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

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        if n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)

        if self.n_cols == 1:
            # TODO: Throw algorithm when this changes algorithm from the user's
            # choice. Github issue #602
            self.algo = 0

        self.n_alpha = 1

        self.coef_ = cudf.Series(zeros(self.n_cols,
                                       dtype=self.dtype))
        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)

        cdef float c_intercept1
        cdef double c_intercept2
        cdef float c_alpha1
        cdef double c_alpha2
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            c_alpha1 = self.alpha
            ridgeFit(handle_[0],
                     <float*>X_ptr,
                     <int>n_rows,
                     <int>self.n_cols,
                     <float*>y_ptr,
                     <float*>&c_alpha1,
                     <int>self.n_alpha,
                     <float*>coef_ptr,
                     <float*>&c_intercept1,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo)

            self.intercept_ = c_intercept1
        else:
            c_alpha2 = self.alpha

            ridgeFit(handle_[0],
                     <double*>X_ptr,
                     <int>n_rows,
                     <int>self.n_cols,
                     <double*>y_ptr,
                     <double*>&c_alpha2,
                     <int>self.n_alpha,
                     <double*>coef_ptr,
                     <double*>&c_intercept2,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo)

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
        X_m, X_ptr, n_rows, n_cols, dtype = \
            input_to_dev_array(X, check_dtype=self.dtype)

        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)
        preds = cudf.Series(zeros(n_rows, dtype=dtype))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            ridgePredict(handle_[0],
                         <float*>X_ptr,
                         <int>n_rows,
                         <int>n_cols,
                         <float*>coef_ptr,
                         <float>self.intercept_,
                         <float*>preds_ptr)
        else:
            ridgePredict(handle_[0],
                         <double*>X_ptr,
                         <int>n_rows,
                         <int>n_cols,
                         <double*>coef_ptr,
                         <double>self.intercept_,
                         <double*>preds_ptr)

        self.handle.sync()

        del(X_m)

        return preds

    def get_params(self, deep=True):
        """
        Sklearn style return parameter state

        Parameters
        -----------
        deep : boolean (default = True)
        """
        params = dict()
        variables = ['alpha', 'fit_intercept', 'normalize', 'solver']
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Sklearn style set parameter state to dictionary of params.

        Parameters
        -----------
        params : dict of new params
        """
        if not params:
            return self
        variables = ['alpha', 'fit_intercept', 'normalize', 'solver']
        for key, value in params.items():
            if key not in variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
        if 'solver' in params.keys():
            self.algo = self._get_algorithm_int(self.solver)
        return self
