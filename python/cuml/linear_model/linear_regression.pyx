#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import warnings

from numba import cuda
from collections import defaultdict

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_cuml_array

cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

    cdef void olsFit(cumlHandle& handle,
                     float *input,
                     int n_rows,
                     int n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     bool normalize, int algo) except +

    cdef void olsFit(cumlHandle& handle,
                     double *input,
                     int n_rows,
                     int n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     bool normalize, int algo) except +

    cdef void olsPredict(cumlHandle& handle,
                         const float *input,
                         int n_rows,
                         int n_cols,
                         const float *coef,
                         float intercept,
                         float *preds) except +

    cdef void olsPredict(cumlHandle& handle,
                         const double *input,
                         int n_rows,
                         int n_cols,
                         const double *coef,
                         double intercept,
                         double *preds) except +


class LinearRegression(Base):

    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    cuML's LinearRegression expects either a cuDF DataFrame or a NumPy matrix
    and provides 2 algorithms SVD and Eig to fit a linear model. SVD is more
    stable, but Eig (default) is much faster.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf

        # Both import methods supported
        from cuml import LinearRegression
        from cuml.linear_model import LinearRegression

        lr = LinearRegression(fit_intercept = True, normalize = False,
                              algorithm = "eig")

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        reg = lr.fit(X,y)
        print("Coefficients:")
        print(reg.coef_)
        print("Intercept:")
        print(reg.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = lr.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 1.0000001
                    1 1.9999998

        Intercept:
                    3.0

        Predictions:

                    0 15.999999
                    1 14.999999

    Parameters
    -----------
    algorithm : 'eig' or 'svd' (default = 'eig')
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        This parameter is ignored when `fit_intercept` is set to False.
        If True, the predictors in X will be normalized by dividing by it's
        L2 norm.
        If False, no scaling will be done.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept_` is False, will be 0.

    Notes
    ------
    LinearRegression suffers from multicollinearity (when columns are
    correlated with each other), and variance explosions from outliers.
    Consider using Ridge Regression to fix the multicollinearity problem, and
    consider maybe first DBSCAN to remove the outliers, or statistical analysis
    to filter possible outliers.

    **Applications of LinearRegression**

        LinearRegression is used in regression tasks where one wants to predict
        say sales or house prices. It is also used in extrapolation or time
        series tasks, dynamic systems modelling and many other machine learning
        tasks. This model should be first tried if the machine learning problem
        is a regression task (predicting a continuous variable).

    For additional information, see `scikitlearn's OLS documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

    For an additional example see `the OLS notebook
    <https://github.com/rapidsai/notebooks/blob/branch-0.12/cuml/linear_regression_demo.ipynb>`_.


    """

    def __init__(self, algorithm='eig', fit_intercept=True, normalize=False,
                 handle=None, verbose=False, output_type=None):
        super(LinearRegression, self).__init__(handle=handle, verbose=verbose,
                                               output_type=output_type)

        # internal array attributes
        self._coef_ = None  # accessed via estimator.coef_
        self._intercept_ = None  # accessed via estimator.intercept_

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        if algorithm in ['svd', 'eig']:
            self.algorithm = algorithm
            self.algo = self._get_algorithm_int(algorithm)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        self.intercept_value = 0.0

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1
        }[algorithm]

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

        self._set_output_type(X)

        cdef uintptr_t X_ptr, y_ptr
        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        X_ptr = X_m.ptr

        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)
        y_ptr = y_m.ptr

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        if n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)

        if self.n_cols == 1 and self.algo != 0:
            warnings.warn("Changing solver from 'eig' to 'svd' as eig " +
                          "solver does not support training data with 1 " +
                          "column currently.", UserWarning)
            self.algo = 0

        self._coef_ = CumlArray.zeros(self.n_cols, dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float c_intercept1
        cdef double c_intercept2
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:

            olsFit(handle_[0],
                   <float*>X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <float*>y_ptr,
                   <float*>coef_ptr,
                   <float*>&c_intercept1,
                   <bool>self.fit_intercept,
                   <bool>self.normalize,
                   <int>self.algo)

            self.intercept_ = c_intercept1
        else:
            olsFit(handle_[0],
                   <double*>X_ptr,
                   <int>n_rows,
                   <int>self.n_cols,
                   <double*>y_ptr,
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

    def predict(self, X, convert_dtype=False):
        """
        Predicts `y` values for `X`.

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
        -------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """

        out_type = self._get_output_type(X)

        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        cdef uintptr_t coef_ptr = self._coef_.ptr

        preds = CumlArray.zeros(n_rows, dtype=dtype)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            olsPredict(handle_[0],
                       <float*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <float*>coef_ptr,
                       <float>self.intercept_,
                       <float*>preds_ptr)
        else:
            olsPredict(handle_[0],
                       <double*>X_ptr,
                       <int>n_rows,
                       <int>n_cols,
                       <double*>coef_ptr,
                       <double>self.intercept_,
                       <double*>preds_ptr)

        self.handle.sync()

        del(X_m)

        return preds.to_output(out_type)

    def get_param_names(self):
        return ['algorithm', 'fit_intercept', 'normalize']
