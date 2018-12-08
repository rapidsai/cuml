cimport c_glm
import numpy as np
cimport numpy as np
from numba import cuda
import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from c_glm cimport *

class linear_model:

    class LinearRegression:
    
        def __init__(self, algorithm = 'qr', fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
            self.coef_ = None
            self.intercept_ = None
            self.fit_intercept = fit_intercept
            self.normalize = normalize
            self.copy_X = copy_X
            self.n_jobs = n_jobs
            if algorithm in ['svd', 'eig', 'qr', 'cholesky']:
                self.algo = self._get_algorithm_int(algorithm)
            else:
                msg = "algorithm {!r} is not supported"
                raise TypeError(msg.format(algorithm))
 
            self.intercept_value = 0.0

        def _get_algorithm_int(self, algorithm):
            return {
            'svd': 0,
            'eig': 1,
            'qr': 2,
            'cholesky': 3
            }[algorithm]

        def _get_ctype_ptr(self, obj):
            # The manner to access the pointers in the gdf's might change, so
            # encapsulating access in the following 3 methods. They might also be
            # part of future gdf versions.
            return obj.device_ctypes_pointer.value

        def _get_column_ptr(self, obj):
            return self._get_ctype_ptr(obj._column._data.to_gpu_array())


        def fit(self, X, y):
            
            cdef uintptr_t input_ptr
            if (isinstance(X, cudf.DataFrame)):
                self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
                X_m = X.as_gpu_matrix()
                self.n_rows = len(X)
                self.n_cols = len(X._cols)

            elif (isinstance(X, np.ndarray)):
                self.gdf_datatype = X.dtype
                X_m = cuda.to_device(np.array(X, order='F'))
                self.n_rows = X.shape[0]
                self.n_cols = X.shape[1]

            else:
                msg = "X matrix format  not supported"
                raise TypeError(msg)

            input_ptr = self._get_ctype_ptr(X_m)

            cdef uintptr_t labels_ptr
            if (isinstance(y, cudf.Series)):
                labels_ptr = self._get_column_ptr(y)
            elif (isinstance(X, np.ndarray)):
                labels_ptr = self._get_ctype_ptr(cuda.to_device(y))
            else:
                msg = "y format  not supported"
                raise TypeError(msg)

            self.coef_ = cudf.Series(np.zeros(self.n_cols, dtype=self.gdf_datatype))
            cdef uintptr_t coef_ptr = self._get_column_ptr(self.coef_)
            
            cdef float c_intercept1
            cdef double c_intercept2
            if self.gdf_datatype.type == np.float32:
                
                c_glm.olsFit(<float*>input_ptr, 
                             <int>self.n_rows, 
                             <int>self.n_cols, 
                             <float*>labels_ptr, 
                             <float*>coef_ptr,
                             <float*>&c_intercept1,
                             <bool>self.fit_intercept,
                             <bool>self.normalize, 
                             <int>self.algo) 

                self.intercept_ = c_intercept1
            else:
                c_glm.olsFit(<double*>input_ptr,
                             <int>self.n_rows,
                             <int>self.n_cols,
                             <double*>labels_ptr,
                             <double*>coef_ptr,
                             <double*>&c_intercept2,
                             <bool>self.fit_intercept,
                             <bool>self.normalize,
                             <int>self.algo)
            
                self.intercept_ = c_intercept2

            return self
        
        def predict(self, X):

            cdef uintptr_t input_ptr
            if (isinstance(X, cudf.DataFrame)):
                gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
                X_m = X.as_gpu_matrix()
                n_rows = len(X)
                n_cols = len(X._cols)

            elif (isinstance(X, np.ndarray)):
                self.gdf_datatype = X.dtype
                X_m = cuda.to_device(np.array(X, order='F'))
                n_rows = X.shape[0]
                n_cols = X.shape[1]

            else:
                msg = "X matrix format  not supported"
                raise TypeError(msg)

            input_ptr = self._get_ctype_ptr(X_m)

            cdef uintptr_t coef_ptr = self._get_column_ptr(self.coef_)
            preds = cudf.Series(np.zeros(n_rows, dtype=gdf_datatype))
            cdef uintptr_t preds_ptr = self._get_column_ptr(preds)
            
            if self.gdf_datatype.type == np.float32:
                c_glm.olsPredict(<float*>input_ptr, 
                                 <int>n_rows,
                                 <int>n_cols, 
                                 <float*>coef_ptr, 
                                 <float>self.intercept_,
                                 <float*>preds_ptr)
            else:
                c_glm.olsPredict(<double*>input_ptr, 
                                 <int>n_rows,
                                 <int>n_cols, 
                                 <double*>coef_ptr, 
                                 <double>self.intercept_,
                                 <double*>preds_ptr)

            return preds
