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

cdef extern from "solver/solver_c.h" namespace "ML::Solver":

    cdef void cdFit(float *input,
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
		   float tol)

    
    cdef void cdFit(double *input,
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
		   double tol)
 
    cdef void cdPredict(const float *input, 
                         int n_rows, 
                         int n_cols, 
                         const float *coef,
                         float intercept, 
                         float *preds,
                         int loss)

    cdef void cdPredict(const double *input, 
                         int n_rows, 
                         int n_cols,
                         const double *coef, 
                         double intercept, 
                         double *preds,
                         int loss)                    

class CD:
    """
    Coordinate Descent (CD) is a very common optimization algorithm that minimizes along 
    coordinate directions to find the minimum of a function.

    cuML's CD algorithm accepts a numpy matrix or a cuDF DataFrame as the input dataset.
    The CD algorithm currently works with linear regression and ridge, lasso, and elastic-net penalties.

    Examples
    ---------

    .. code-block:: python

        import numpy as np
        import cudf
        from cuml.solvers import CD as cumlCD

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)
        y = cudf.Series(np.array([1, 1, 2, 2], dtype=np.float32))
        pred_data = cudf.DataFrame()
        pred_data['col1'] = np.asarray([3, 2], dtype=datatype)
        pred_data['col2'] = np.asarray([5, 5], dtype=datatype)

        cu_cd = cumlCD(max_iter=2000, fit_intercept=True,
                        tol=0.0, penalty=penalty, loss=loss)

        cu_cd.fit(X, y)
        cu_pred = cu_cd.predict(pred_data).to_array()
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
       'l1' performs L1 norm (Lasso) which minimizes the sum of the abs value of coefficients
       'l2' performs L2 norm (Ridge) which minimizes the sum of the square of the coefficients
       'elasticnet' performs Elastic Net regularization which is a weighted average of L1 and L2 norms
    alpha: float (default = 0.0001)
        The constant value which decides the degree of regularization
    fit_intercept : boolean (default = True)
       If True, the model tries to correct for the global mean of y.
       If False, the model expects that you have centered the data.        
    max_iter : int (default = 1000)
        The number of times the model should iterate through the entire dataset during training (default = 1000)
    tol : float (default = 1e-3)
       The training process will stop if current_loss > previous_loss - tol 
    shuffle : boolean (default = True)
       True, shuffles the training data after each epoch
       False, does not shuffle the training data after each epoch
    
    """
    
    def __init__(self, loss='squared_loss', alpha=0.0001, l1_ratio=0.15, 
        fit_intercept=True, normalize=False, max_iter=1000, tol=1e-3, shuffle=True):
        
        if loss in ['squared_loss']:
            self.loss = self._get_loss_int(loss)
        else:
            msg = "loss {!r} is not supported"
            raise TypeError(msg.format(loss))

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
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

        cdef float c_intercept1 
        cdef double c_intercept2
        
        if self.gdf_datatype.type == np.float32:
            cdFit(<float*>X_ptr, 
                       <int>self.n_rows, 
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
            cdFit(<double*>X_ptr, 
                       <int>self.n_rows, 
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
            cdPredict(<float*>X_ptr,
                           <int>n_rows,
                           <int>n_cols,
                           <float*>coef_ptr,
                           <float>self.intercept_,
                           <float*>preds_ptr,
                           <int>self.loss)
        else:
            cdPredict(<double*>X_ptr,
                           <int>n_rows,
                           <int>n_cols,
                           <double*>coef_ptr,
                           <double>self.intercept_,
                           <double*>preds_ptr,
                           <int>self.loss)

        del(X_m)

        return preds

