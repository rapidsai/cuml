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

cimport svm
import numpy as np
cimport numpy as np
from numba import cuda
import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t


class SVC:
    def __init__(self, tol=1e-3, C=1):

        self.tol = tol
        self.C = C

    def _get_kernel_int(self, loss):
        return {
            'rbf': 0,
            'whatever': 1,
            'linear': 2,
        }[loss]

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

        self.n_alpha = 1

        self.coef_ = cudf.Series(np.zeros(self.n_cols, dtype=self.gdf_datatype))
        cdef uintptr_t coef_ptr = self._get_column_ptr(self.coef_)

        
        if self.gdf_datatype.type == np.float32:
            svm.svcFit(<float*>X_ptr, 
                       <int>self.n_rows, 
                       <int>self.n_cols, 
                       <float*>y_ptr, 
                       <float*>coef_ptr,
                       <float>self.C, 
                       <float>self.tol)
        else:
            svm.svcFit(<double*>X_ptr, 
                       <int>self.n_rows, 
                       <int>self.n_cols, 
                       <double*>y_ptr, 
                       <double*>coef_ptr,
                       <double>self.C,
                       <double>self.tol)
            
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


        #if pred_datatype.type == np.float32:
            #sgd.sgdPredict(<float*>X_ptr,
                           #<int>n_rows,
                           #<int>n_cols,
                           #<float*>coef_ptr,
                           #<float>self.intercept_,
                           #<float*>preds_ptr,
                           #<int>self.loss)
        #else:
            #sgd.sgdPredict(<double*>X_ptr,
                           #<int>n_rows,
                           #<int>n_cols,
                           #<double*>coef_ptr,
                           #<double>self.intercept_,
                           #<double*>preds_ptr,
                           #<int>self.loss)

        #del(X_m)

        return preds
