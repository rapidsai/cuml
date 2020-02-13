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
import cupy
import numpy as np

from numba import cuda

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

from cuml.common.base import Base
from cuml.metrics import r2_score
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_dev_array, zeros, get_cudf_column_ptr, \
    device_array_from_ptr, get_dev_array_ptr
from libcpp cimport bool
from cuml.svm.svm_base import SVMBase

cdef extern from "cuml/matrix/kernelparams.h" namespace "MLCommon::Matrix":
    enum KernelType:
        LINEAR, POLYNOMIAL, RBF, TANH

    cdef struct KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0

cdef extern from "cuml/svm/svm_parameter.h" namespace "ML::SVM":
    enum SvmType:
        C_SVC, NU_SVC, EPSILON_SVR, NU_SVR

    cdef struct svmParameter:
        # parameters for trainig
        double C
        double cache_size
        int max_iter
        int nochange_steps
        double tol
        int verbose
        double epsilon
        SvmType svmType

cdef extern from "cuml/svm/svm_model.h" namespace "ML::SVM":
    cdef cppclass svmModel[math_t]:
        # parameters of a fitted model
        int n_support
        int n_cols
        math_t b
        math_t *dual_coefs
        math_t *x_support
        int *support_idx
        int n_classes
        math_t *unique_labels

cdef extern from "cuml/svm/svc.hpp" namespace "ML::SVM":

    cdef void svcFit[math_t](const cumlHandle &handle, math_t *input,
                             int n_rows, int n_cols, math_t *labels,
                             const svmParameter &param,
                             KernelParams &kernel_params,
                             svmModel[math_t] &model) except+

    cdef void svcPredict[math_t](
        const cumlHandle &handle, math_t *input, int n_rows, int n_cols,
        KernelParams &kernel_params, const svmModel[math_t] &model,
        math_t *preds, math_t buffer_size, bool predict_class) except +

    cdef void svmFreeBuffers[math_t](const cumlHandle &handle,
                                     svmModel[math_t] &m) except +

cdef extern from "cuml/svm/svr.hpp" namespace "ML::SVM":

    cdef void svrFit[math_t](const cumlHandle &handle, math_t *X,
                             int n_rows, int n_cols, math_t *y,
                             const svmParameter &param,
                             KernelParams &kernel_params,
                             svmModel[math_t] &model) except+


class SVR(SVMBase):
    """
    SVR (Epsilon Support Vector Regression)

    The solver uses the SMO method to fit the regressor. We use the Optimized
    Hierarchical Decomposition [1] variant of the SMO algorithm, similar to [2]

    References
    ----------
    [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical Decomposition
         Algorithm for Support VectorMachine Training, IEEE Transactions on
         Parallel and Distributed Systems, vol 28, no 12, 3330, (2017)
    [2] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
    *      of Machine Learning Research, 19, 1-5 (2018)
        https://github.com/Xtra-Computing/thundersvm

    """
    def __init__(self, handle=None, C=1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, epsilon=0.1,
                 cache_size=200.0, max_iter=-1, nochange_steps=1000,
                 verbose=False):
        """
        Construct an SVC classifier for training and predictions.

        Parameters
        ----------
        handle : cuml.Handle
            If it is None, a new one is created for this class
        C : float (default = 1.0)
            Penalty parameter C
        kernel : string (default='rbf')
            Specifies the kernel function. Possible options: 'linear', 'poly',
            'rbf', 'sigmoid'. Currently precomputed kernels are not supported.
        degree : int (default=3)
            Degree of polynomial kernel function.
        gamma : float or string (default = 'scale')
            Coefficient for rbf, poly, and sigmoid kernels. You can specify the
            numeric value, or use one of the following options:
            - 'auto': gamma will be set to 1 / n_features
            - 'scale': gamma will be se to 1 / (n_features * X.var())
        coef0 : float (default = 0.0)
            Independent term in kernel function, only signifficant for poly and
            sigmoid
        tol : float (default = 1e-3)
            Tolerance for stopping criterion.
        epsilon: float (default = 0.1)
            epsilon parameter of the epsiron-SVR model. There is no penalty
            associated to points that are predicted within the epsilon-tube
            around the target values.
        cache_size : float (default = 200 MiB)
            Size of the kernel cache during training in MiB. The default is a
            conservative value, increase it to improve the training time, at
            the cost of higher memory footprint. After training the kernel
            cache is deallocated.
            During prediction, we also need a temporary space to store kernel
            matrix elements (this can be signifficant if n_support is large).
            The cache_size variable sets an upper limit to the prediction
            buffer as well.
        max_iter : int (default = 100*n_samples)
            Limit the number of outer iterations in the solver
        nochange_steps : int (default = 1000)
            We monitor how much our stopping criteria changes during outer
            iterations. If it does not change (changes less then 1e-3*tol)
            for nochange_steps consecutive steps, then we stop training.
        verbose : bool (default = False)
            verbose mode

        Attributes
        ----------
        n_support_ : int
            The total number of support vectors. Note: this will change in the
            future to represent number support vectors for each class (like
            in Sklearn, see Issue #956)
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
            Only available for linear kernels. It is the normal of the
            hyperplane.
            coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]

        For additional docs, see `scikitlearn's SVR
        <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.
        """
        super(SVR, self).__init__(handle, C, kernel, degree, gamma, coef0, tol,
                                  cache_size, max_iter, nochange_steps,
                                  verbose, epsilon)
        self.svmType = EPSILON_SVR

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

        self._dealloc()  # delete any previously fitted model
        self._coef_ = None

        cdef KernelParams _kernel_params = self._get_kernel_params(X_m)
        cdef svmParameter param = self._get_svm_params()
        cdef svmModel[float] *model_f
        cdef svmModel[double] *model_d
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            model_f = new svmModel[float]()
            svrFit(handle_[0], <float*>X_ptr, <int>self.n_rows,
                   <int>self.n_cols, <float*>y_ptr, param, _kernel_params,
                   model_f[0])
            self._model = <uintptr_t>model_f
        elif self.dtype == np.float64:
            model_d = new svmModel[double]()
            svrFit(handle_[0], <double*>X_ptr, <int>self.n_rows,
                   <int>self.n_cols, <double*>y_ptr, param, _kernel_params,
                   model_d[0])
            self._model = <uintptr_t>model_d
        else:
            raise TypeError('Input data type should be float32 or float64')

        self._unpack_model()
        self.fit_status_ = 0
        self.handle.sync()

        del X_m
        del y_m

        return self

    def predict(self, X):
        """
        Predicts the values for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
        y : cuDF Series
           Dense vector (floats or doubles) of shape (n_samples, 1)
        """

        return super(SVR, self).predict(X, False)

    def score(self, X, y):
        """
        Return R^2 score of the prediction.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (floats or doubles) of target values.
            Acceptable formats: cuDF Series, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        -------
        score: float R^2 score
        """

        y_hat = self.predict(X)
        return r2_score(y, y_hat)
