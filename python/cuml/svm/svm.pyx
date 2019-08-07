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

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_dev_array, zeros, get_cudf_column_ptr, \
    device_array_from_ptr

from sklearn.exceptions import NotFittedError

cdef extern from "gram/kernelparams.h" namespace "MLCommon::GramMatrix":
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
        math_t *x_support
        int *support_idx
        math_t b
        KernelParams _kernel_params
        math_t C
        math_t tol

        CppSVC(cumlHandle& handle, math_t C, math_t tol,
               KernelParams _kernel_params, float cache_size,
               int max_iter) except+
        void fit(math_t *input, int n_rows, int n_cols, math_t *labels) except+
        void predict(math_t *input, int n_rows, int n_cols,
                     math_t *preds) except+


class SVC(Base):
    """
    SVC (C-Support Vector Classification)

    Currently only binary classification is supported.

    The solver uses the SMO method similarily to fit the classifier.

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
    support_ : int, shape = [n_support]
        Device array of suppurt vector indices
    support_vectors_ : float, shape [n_support, n_cols]
        Device array of support vectors
    dual_coef_ : float, shape = [1, n_support]
        Device array of coefficients for support vectors
    intercept_ : int
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    coef_ : float, shape [1, n_cols]
        Only available for linear kernels. It is the normal of the hyperplane.
        coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """
    def __init__(self, handle=None, C=1, kernel='linear', degree=3,
                 gamma='auto', coef0=0.0, tol=1e-3, cache_size=200.0,
                 max_iter=-1, verbose=False):
        super(SVC, self).__init__(handle=handle, verbose=verbose)
        self.tol = tol
        self.C = C
        self.kernel = kernel
        self._c_kernel = self._get_c_kernel(kernel)
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.dual_coef_ = None
        self.support_ = None
        self.support_vectors_ = None
        self.intercept_ = None
        self.n_support_ = None
        self._coef_ = None
        self.dtype = None
        self._kernel_params = None
        self._svcHandle = None
        self.verbose = verbose
        # The current implementation stores the pointer to CppSVC in svcHandle.
        # The pointer is stored with type size_t (see fit()), because we
        # cannot have cdef CppSVC[float, float] *svc here.
        #
        # CppSVC is not yet created here because the data type will be only
        # known when we call fit(). Similarily kernel_params is not yet
        # initialized here, because gamma can depend on the input data.

    def __dealloc__(self):
        # deallocate CppSVC
        cdef CppSVC[float]* svc_f
        cdef CppSVC[double]* svc_d
        cdef KernelParams *_kernel_params
        if self._svcHandle is not None:
            if self.dtype == np.float32:
                svc_f = <CppSVC[float]*><size_t> self._svcHandle
                del svc_f
            elif self.dtype == np.float64:
                svc_d = <CppSVC[double]*><size_t> self._svcHandle
                del svc_d
            else:
                raise TypeError("Unknown type for SVC class")
        if self._kernel_params is not None:
            _kernel_params = <KernelParams*><size_t> self._kernel_params
            del _kernel_params
        self._svcHandle = None
        self._kernel_params = None

    def _get_c_kernel(self, kernel):
        return {
            'linear': LINEAR,
            'poly': POLYNOMIAL,
            'rbf': RBF,
            'sigmoid': TANH
        }[kernel]

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

    def _calc_coef(self):
        return np.dot(self.dual_coef_.copy_to_host(),
                      self.support_vectors_.copy_to_host())

    @property
    def coef_(self):
        if self._c_kernel != LINEAR:
            raise AttributeError("coef_ is only available for linear kernels")
        if self._svcHandle is None:
            raise NotFittedError("Call fit before prediction")
        if self._coef_ is None:
            self._coef_ = self._calc_coef()
        return self._coef_

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

        X_m, X_ptr, self.n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        y_m, y_ptr, _, _, _ = input_to_dev_array(y,
                                                 convert_to_dtype=self.dtype)

        self.__dealloc__()  # delete any previously allocated CppSVC instance
        self._coef_ = None

        cdef CppSVC[float]* svc_f = NULL
        cdef CppSVC[double]* svc_d = NULL
        cdef KernelParams *_kernel_params
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        _kernel_params = new KernelParams(
            <KernelType>self._c_kernel, <int>self.degree,
            <double> self._gamma_val(X), <double>self.coef0)
        self._kernel_params = <size_t> _kernel_params

        if self.dtype == np.float32:
            svc_f = new CppSVC[float](
                handle_[0], self.C, self.tol, deref(_kernel_params),
                self.cache_size, self.max_iter)
            self._svcHandle = <size_t> svc_f
            svc_f.fit(<float*>X_ptr, <int>self.n_rows,
                      <int>self.n_cols, <float*>y_ptr)
            self.intercept_ = svc_f.b
            self.n_support_ = svc_f.n_support
            self.dual_coef_ = device_array_from_ptr(
                <uintptr_t>svc_f.dual_coefs, (1, self.n_support_), self.dtype)
            self.support_ = device_array_from_ptr(
                <uintptr_t>svc_f.support_idx, (self.n_support_,), np.int32)
            self.support_vectors_ = device_array_from_ptr(
                <uintptr_t>svc_f.x_support, (self.n_support_, self.n_cols),
                self.dtype)
            self.fit_status_ = 0
        elif self.dtype == np.float64:
            svc_d = new CppSVC[double](
                handle_[0], self.C, self.tol, deref(_kernel_params),
                self.cache_size, self.max_iter)
            self._svcHandle = <size_t> svc_d
            svc_d.fit(<double*>X_ptr, <int>self.n_rows, <int>self.n_cols,
                      <double*>y_ptr)
            self.intercept_ = svc_d.b
            self.n_support_ = svc_d.n_support
            self.dual_coef_ = device_array_from_ptr(
                <uintptr_t>svc_d.dual_coefs, (1, self.n_support_), self.dtype)
            self.support_ = device_array_from_ptr(
                <uintptr_t>svc_d.support_idx, (self.n_support_,), np.int32)
            self.support_vectors_ = device_array_from_ptr(
                <uintptr_t>svc_d.x_support, (self.n_support_, self.n_cols),
                self.dtype)
            self.fit_status_ = 0
        else:
            raise TypeError('Input data type should be float32 or float64')

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
        y: cuDF Series
           Dense vector (floats or doubles) of shape (n_samples, 1)
        """

        if self._svcHandle is None:
            raise NotFittedError("Call fit before prediction")

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, pred_dtype = \
            input_to_dev_array(X, check_dtype=self.dtype)

        preds = cudf.Series(zeros(n_rows, dtype=self.dtype))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)

        cdef CppSVC[float]* svc_f
        cdef CppSVC[double]* svc_d

        if self.dtype == np.float32:
            svc_f = <CppSVC[float]*><size_t> self._svcHandle
            svc_f.predict(<float*>X_ptr,
                          <int>n_rows,
                          <int>n_cols,
                          <float*>preds_ptr)
        else:
            svc_d = <CppSVC[double]*><size_t> self._svcHandle
            svc_d.predict(<double*>X_ptr,
                          <int>n_rows,
                          <int>n_cols,
                          <double*>preds_ptr)

        self.handle.sync()

        del(X_m)

        return preds
