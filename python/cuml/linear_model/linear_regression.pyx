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

from numba import cuda
from collections import defaultdict
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from sklearn.utils.fixes import signature

cdef extern from "glm/glm_c.h" namespace "ML::GLM":

    cdef void olsFit(float *input,
                     int n_rows,
                     int n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     bool normalize, int algo)

    cdef void olsFit(double *input,
                     int n_rows,
                     int n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     bool normalize, int algo)

    cdef void olsPredict(const float *input,
                         int n_rows,
                         int n_cols,
                         const float *coef,
                         float intercept,
                         float *preds)

    cdef void olsPredict(const double *input,
                         int n_rows,
                         int n_cols,
                         const double *coef,
                         double intercept,
                         double *preds)


class LinearRegression:

    """
    LinearRegression is a simple machine learning model where the response y is modelled by a 
    linear combination of the predictors in X.

    cuML's LinearRegression expects either a cuDF DataFrame or a NumPy matrix and provides 2 
    algorithms SVD and Eig to fit a linear model. SVD is more stable, but Eig (default) 
    is much more faster.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import cudf

        # Both import methods supported
        from cuml import LinearRegression
        from cuml.linear_model import LinearRegression

        lr = LinearRegression(fit_intercept = True, normalize = False, algorithm = "eig")

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        reg = lr.fit(X,y)
        print("Coefficients:")
        print(reg.coef_)
        print("intercept:")
        print(reg.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = lr.predict(X_new)

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
    ----------
    algorithm : 'eig' or 'svd' (default = 'eig')
        Eig uses a eigendecomposition of the covariance matrix, and is much faster.
        SVD is slower, but is guaranteed to be stable.
    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2 norm.
        If False, no scaling will be done.

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If fit_intercept_ is False, will be 0.


    For additional docs, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

    """

    def __init__(self, algorithm='eig', fit_intercept=True, normalize=False):

        """
        Initializes the linear regression class.

        Parameters
        ----------
        algorithm : Type: string. 'eig' (default) and 'svd' are supported algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        if algorithm in ['svd', 'eig']:
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

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

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

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg) 

        if self.n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)          

        if self.n_cols == 1:
            self.algo = 0 # eig based method doesn't work when there is only one column.

        X_ptr = self._get_ctype_ptr(X_m)

        cdef uintptr_t y_ptr
        if (isinstance(y, cudf.Series)):
            y_ptr = self._get_column_ptr(y)
        elif (isinstance(y, np.ndarray)):
            y_m = cuda.to_device(y)
            y_ptr = self._get_ctype_ptr(y_m)
        else:
            msg = "y vector must be a cuDF series or Numpy ndarray"
            raise TypeError(msg)

        self.coef_ = cudf.Series(np.zeros(self.n_cols, dtype=self.gdf_datatype))
        cdef uintptr_t coef_ptr = self._get_column_ptr(self.coef_)

        cdef float c_intercept1
        cdef double c_intercept2

        if self.gdf_datatype.type == np.float32:

            olsFit(<float*>X_ptr,
                       <int>self.n_rows,
                       <int>self.n_cols,
                       <float*>y_ptr,
                       <float*>coef_ptr,
                       <float*>&c_intercept1,
                       <bool>self.fit_intercept,
                       <bool>self.normalize,
                       <int>self.algo)

            self.intercept_ = c_intercept1
        else:
            olsFit(<double*>X_ptr,
                       <int>self.n_rows,
                       <int>self.n_cols,
                       <double*>y_ptr,
                       <double*>coef_ptr,
                       <double*>&c_intercept2,
                       <bool>self.fit_intercept,
                       <bool>self.normalize,
                       <int>self.algo)

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

        X_ptr = self._get_ctype_ptr(X_m)

        cdef uintptr_t coef_ptr = self._get_column_ptr(self.coef_)
        preds = cudf.Series(np.zeros(n_rows, dtype=pred_datatype))
        cdef uintptr_t preds_ptr = self._get_column_ptr(preds)

        if pred_datatype.type == np.float32:
            olsPredict(<float*>X_ptr,
                           <int>n_rows,
                           <int>n_cols,
                           <float*>coef_ptr,
                           <float>self.intercept_,
                           <float*>preds_ptr)
        else:
            olsPredict(<double*>X_ptr,
                           <int>n_rows,
                           <int>n_cols,
                           <double*>coef_ptr,
                           <double>self.intercept_,
                           <double*>preds_ptr)

        del(X_m)

        return preds

      
    def get_params(self, deep=True):
        params = dict()
        variables = ['algorithm','fit_intercept','normalize']
        for key in variables:
            var_value = getattr(self,key,None)
            params[key] = var_value   
        return params

      
    def set_params(self, **params):
        if not params:
            return self
        variables = ['algorithm','fit_intercept','normalize']

        current_params = {"algorithm" : self.algo,"fit_intercept" : self.fit_intercept,"normalize" : self.normalize}
        for key, value in params.items():
            if key not in current_params:
                raise ValueError('Invalid parameter %s for estimator')
            else:
                setattr(self, key, value)
                current_params[key] = value
        if params["algorithm"]=='eig':
            self.algo=1
        else:
            self.algo = 0
        return self

