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

import numpy as np
cimport numpy as np
from numba import cuda

import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from sklearn.exceptions import NotFittedError
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
#from cuml.decomposition.utils cimport *

cdef extern from "svm/kernelparams.h" namespace "ML::SVM":
    enum KernelType:
        LINEAR, POLYNOMIAL, RBF, TANH

    cdef cppclass KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0
        KernelParams(KernelType kernel, int degree, double gamma, double coef0)

cdef extern from "svm/svc.h" namespace "ML::SVM":

    cdef cppclass CppSVC "ML::SVM::SVC" [math_t]:
        # The CppSVC class manages the memory of the parameters that are found
        # during fitting (the support vectors, and the dual_coefficients). The
        # number of these parameters are not known before fitting.
        int n_support
        math_t *dual_coefs
        int *support_idx
        math_t b
        KernelParams kernel_params
        math_t C
        math_t tol

        CppSVC(cumlHandle& handle, math_t C, math_t tol,
            KernelParams kernel_params, float cache_size, int max_iter) except+
        void fit(math_t *input, int n_rows, int n_cols, math_t *labels) except+
        void predict(math_t *input, int n_rows, int n_cols, math_t *preds) \
            except+


class SVC(Base):
    """
    SVC (C-Support Vector Classification)

    Currently only binary classification is supported.

    Parameters
    ----------
    handle : cuml.Handle
        If it is None, a new one is created for this class
    C : float (default = 1.0)
        Penalty parameter C
    kernel : string (default='linear' TODO change to 'rbf' once implemented)
        Specifies the kernel function. Possible options: 'linear', 'poly',
        'rbf', 'sigmoid', 'precomputed'
    degree : int (default=3)
        Degree of polynomial kernel function.
    gamma : float (default = 'auto')
        Coefficient for rbf, poly, and sigmoid kernels.
    coef0 : float (default = 0.0)
        Independent term in kernel function, only signifficant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 200 MiB)
        Size of the kernel cache n MiB (TODO)
    max_iter : int (default = 100*n_samples)
        Limit the number of outer iterations in the solver
    verbose : bool (default = False)
        verbose mode

    Attributes
    ----------

    n_support_ : int
        The total number of support vectors.
        TODO change this to represent number support vectors for each class.
    intercept_ : int
        The constant in the decision function

    The solver uses the SMO method similarily to ThunderSVM and OHD-SVM.
    """
    def __init__(self, handle=None, C=1, kernel='linear', degree=3,
        gamma='auto', coef0=0.0, tol=1e-3, cache_size = 200.0, max_iter = -1,
        verbose=False):
        super(SVC, self).__init__(handle=handle, verbose=verbose)
        self.tol = tol
        self.C = C
        self.kernel = self._get_c_kernel(kernel)
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.dual_coefs_ = None # TODO populate this after fitting
        self.intercept_ = None
        self.n_support_ = None
        self.gdf_datatype = None
        self.kernel_params = None
        self.svcHandle = None
        self.verbose = verbose
        # The current implementation stores the pointer to CppSVC in svcHandle.
        # The pointer is stored with type size_t (see fit()), because we
        # cannot have cdef CppSVC[float, float] *svc here. This leads to ugly
        # two step conversion when we want to use the actual pointer.
        #
        # Alternatively we could have a thin cdef class PyCppSVC wrapper around
        # CppSVC, or we could make SVC itself a cdef class.
        #
        # CppSVC is not yet created here because the data type will be only
        # known when we call fit(). Similarily kernel_params is not yet created
        # here, because gamma can depend on the input data.

    def __dealloc__(self):
        # deallocate CppSVC
        cdef CppSVC[float]* svc_f
        cdef CppSVC[double]* svc_d
        cdef KernelParams *kernel_params
        #cdef size_t p = self.svcHandle
        if self.svcHandle is not None:
            if self.gdf_datatype.type == np.float32:
                svc_f = <CppSVC[float]*><size_t> self.svcHandle
                del svc_f
            elif self.gdf_datatype.type == np.float64:
                svc_d = <CppSVC[double]*><size_t> self.svcHandle
                del svc_d
            else:
                raise TypeError("Unknown type for SVC class")
        if self.kernel_params is not None:
            #p = self.kernel_params
            kernel_params = <KernelParams*><size_t> self.kernel_params
            del kernel_params


    def _get_c_kernel(self, kernel):
        return {
            'linear': LINEAR,
            'poly': POLYNOMIAL,
            'rbf': RBF,
            'sigmoid': TANH
        } [kernel];

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _gamma_val(self, X):
        # Calculate the value for gamma kernel parameter
        if type(self.gamma) is str:
            if self.gamma == 'auto':
                return 1 / self.n_cols
            elif self.gamma == 'scale':
                x_std = X.std()
                return 1 / (self.n_cols * x_std)
            else:
                raise ValueError("Not implemented gamma option: " + self.gamma)
        else:
          return self.gamma

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Parameters
        ----------
        X : cuDF DataFrame or NumPy array
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        y: cuDF DataFrame or NumPy array
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

        cdef CppSVC[float]* svc_f = NULL
        cdef CppSVC[double]* svc_d = NULL

        cdef KernelParams *kernel_params
        if self.kernel_params is None:
            kernel_params = new KernelParams(<KernelType>self.kernel, <int>self.degree,
            <double> self._gamma_val(X), <double>self.coef0)
            self.kernel_params = <size_t> kernel_params
        else:
            kernel_params = <KernelParams*><size_t> self.kernel_params

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.gdf_datatype.type == np.float32:
            if self.svcHandle is None:
                svc_f = new CppSVC[float](handle_[0], self.C, self.tol,
                    deref(kernel_params),
                    self.cache_size, self.max_iter)
                self.svcHandle = <size_t> svc_f
            else:
                svc_f =  <CppSVC[float]*><size_t> self.svcHandle
            svc_f.fit(<float*>X_ptr, <int>self.n_rows,
                      <int>self.n_cols, <float*>y_ptr)
            self.intercept_ = svc_f.b
            self.n_support_ = svc_f.n_support


        else:
            if self.svcHandle is None:
                svc_d = new CppSVC[double](handle_[0], self.C, self.tol,
                    deref(kernel_params),
                    self.cache_size, self.max_iter)
                self.svcHandle = <size_t> svc_d
            else:
                svc_d =  <CppSVC[double]*><size_t> self.svcHandle
            svc_d.fit(<double*>X_ptr, <int>self.n_rows,
                      <int>self.n_cols, <double*>y_ptr)
            self.intercept_ = svc_d.b
            self.n_support_ = svc_d.n_support

        del X_m
        del y_m

        return self

    def predict(self, X):
        """
        Predicts the y for X.

        Parameters
        ----------
        X : cuDF DataFrame or NumPy array
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        ----------
        y: cuDF DataFrame or NumPy array (depending on input type)
           Dense vector (floats or doubles) of shape (n_samples, )
        """

        if self.svcHandle is None:
            raise NotFittedError("Call fit before prediction")

        cdef uintptr_t preds_ptr

        if (isinstance(X, cudf.DataFrame)):
            pred_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='F')
            n_rows = len(X)
            n_cols = len(X._cols)
            preds = cudf.Series(np.zeros(n_rows, dtype=pred_datatype))
            preds_ptr = self._get_column_ptr(preds)

        elif (isinstance(X, np.ndarray)):
            pred_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_rows = X.shape[0]
            n_cols = X.shape[1]
            preds = cuda.device_array((X.shape[0],), dtype=pred_datatype, order='F')
            preds_ptr = self._get_ctype_ptr(preds)

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        if self.gdf_datatype.type != pred_datatype.type:
            msg = "Datatype for prediction should be the same as for fitting"
            raise TypeError(msg)

        cdef uintptr_t X_ptr = self._get_ctype_ptr(X_m)

        cdef CppSVC[float]* svc_f
        cdef CppSVC[double]* svc_d

        if pred_datatype.type == np.float32:
            svc_f = <CppSVC[float]*><size_t> self.svcHandle
            svc_f.predict(<float*>X_ptr,
                          <int>n_rows,
                          <int>n_cols,
                          <float*>preds_ptr)
        else:
            svc_d = <CppSVC[double]*><size_t> self.svcHandle
            svc_d.predict(<double*>X_ptr,
                          <int>n_rows,
                          <int>n_cols,
                          <double*>preds_ptr)

        if isinstance(X, np.ndarray):
            preds = preds.copy_to_host()

        del(X_m)

        return preds
