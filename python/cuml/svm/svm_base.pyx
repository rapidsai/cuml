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
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_dev_array, zeros, get_cudf_column_ptr, \
    device_array_from_ptr, get_dev_array_ptr
from libcpp cimport bool

cdef extern from "cuml/matrix/kernelparams.h" namespace "MLCommon::Matrix":
    enum KernelType:
        LINEAR,
        POLYNOMIAL,
        RBF,
        TANH

    cdef struct KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0

cdef extern from "cuml/svm/svm_parameter.h" namespace "ML::SVM":
    enum SvmType:
        C_SVC,
        NU_SVC,
        EPSILON_SVR,
        NU_SVR

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


class SVMBase(Base):
    """
    Base class for Support Vector Machines

    Currently only binary classification is supported.

    The solver uses the SMO method to fit the classifier. We use the Optimized
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
                 gamma='auto', coef0=0.0, tol=1e-3, cache_size=200.0,
                 max_iter=-1, nochange_steps=1000, verbose=False,
                 epsilon=0.1):
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
        gamma : float or string (default = 'auto')
            Coefficient for rbf, poly, and sigmoid kernels. You can specify the
            numeric value, or use one of the following options:
            - 'auto': gamma will be set to 1 / n_features
            - 'scale': gamma will be se to 1 / (n_features * X.var())
        coef0 : float (default = 0.0)
            Independent term in kernel function, only signifficant for poly and
            sigmoid
        tol : float (default = 1e-3)
            Tolerance for stopping criterion.
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

        For additional docs, see `scikitlearn's SVC
        <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
        """
        super(SVMBase, self).__init__(handle=handle, verbose=verbose)
        # Input parameters for training
        self.tol = tol
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps
        self.verbose = verbose
        self.epsilon = epsilon
        self.svmType = None  # Child class should set self.svmType

        # Attributes (parameters of the fitted model)
        self.dual_coef_ = None
        self.support_ = None
        self.support_vectors_ = None
        self.intercept_ = None
        self.n_support_ = None

        self._c_kernel = self._get_c_kernel(kernel)
        self._gamma_val = None  # the actual numerical value used for training
        self._coef_ = None  # value of the coef_ attribute, only for lin kernel
        self.dtype = None
        self._model = None  # structure of the model parameters
        self._freeSvmBuffers = False  # whether to call the C++ lib for cleanup

    def __del__(self):
        self._dealloc()

    def _dealloc(self):
        # deallocate model parameters
        cdef svmModel[float] *model_f
        cdef svmModel[double] *model_d
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        if self._model is not None:
            if self.dtype == np.float32:
                model_f = <svmModel[float]*><uintptr_t> self._model
                if self._freeSvmBuffers:
                    svmFreeBuffers(handle_[0], model_f[0])
                del model_f
            elif self.dtype == np.float64:
                model_d = <svmModel[double]*><uintptr_t> self._model
                if self._freeSvmBuffers:
                    svmFreeBuffers(handle_[0], model_d[0])
                del model_d
            else:
                raise TypeError("Unknown type for SVC class")
            try:
                del self.fit_status_
            except AttributeError:
                pass

        self._model = None

    def _get_c_kernel(self, kernel):
        """
        Get KernelType from the kernel string.

        Paramaters
        ----------
        kernel: string, ('linear', 'poly', 'rbf', or 'sigmoid')
        """
        return {
            'linear': LINEAR,
            'poly': POLYNOMIAL,
            'rbf': RBF,
            'sigmoid': TANH
        }[kernel]

    def _calc_gamma_val(self, X):
        """
        Calculate the value for gamma kernel parameter.

        Parameters
        ----------
        X: array like
            Array of training vectors. The 'auto' and 'scale' gamma options
            derive the numerical value of the gamma parameter from X.
        """
        if type(self.gamma) is str:
            if self.gamma == 'auto':
                return 1 / self.n_cols
            elif self.gamma == 'scale':
                x_var = cupy.asarray(X).var()
                return 1 / (self.n_cols * x_var)
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
        if self._model is None:
            raise RuntimeError("Call fit before prediction")
        if self._coef_ is None:
            self._coef_ = self._calc_coef()
        return self._coef_

    def _get_kernel_params(self, X=None):
        """ Wrap the kernel parameters in a KernelParams obtect """
        cdef KernelParams _kernel_params
        if X is not None:
            self._gamma_val = self._calc_gamma_val(X)
        _kernel_params.kernel = self._c_kernel
        _kernel_params.degree = self.degree
        _kernel_params.gamma = self._gamma_val
        _kernel_params.coef0 = self.coef0
        return _kernel_params

    def _get_svm_params(self):
        """ Wrap the training parameters in an svmParameter obtect """
        cdef svmParameter param
        param.C = self.C
        param.cache_size = self.cache_size
        param.max_iter = self.max_iter
        param.nochange_steps = self.nochange_steps
        param.tol = self.tol
        param.verbose = self.verbose
        param.epsilon = self.epsilon
        param.svmType = self.svmType
        return param

    def _get_svm_model(self):
        """ Wrap the fitted model parameters into an svmModel structure.
        This is used if the model is loaded by pickle, the self._model struct
        that we can pass to the predictor.
        """
        cdef svmModel[float] *model_f
        cdef svmModel[double] *model_d
        if self.dual_coef_ is None:
            # the model is not fitted in this case
            return None
        if self.dtype == np.float32:
            model_f = new svmModel[float]()
            model_f.n_support = self.n_support_
            model_f.n_cols = self.n_cols
            model_f.b = self.intercept_
            model_f.dual_coefs = \
                <float*><size_t>get_dev_array_ptr(self.dual_coef_)
            model_f.x_support = \
                <float*><uintptr_t>get_dev_array_ptr(self.support_vectors_)
            model_f.support_idx = \
                <int*><uintptr_t>get_dev_array_ptr(self.support_)
            model_f.n_classes = self._n_classes
            if self._n_classes > 0:
                model_f.unique_labels = \
                    <float*><uintptr_t>get_dev_array_ptr(self._unique_labels)
            else:
                model_f.unique_labels = NULL
            return <uintptr_t>model_f
        else:
            model_d = new svmModel[double]()
            model_d.n_support = self.n_support_
            model_d.n_cols = self.n_cols
            model_d.b = self.intercept_
            model_d.dual_coefs = \
                <double*><size_t>get_dev_array_ptr(self.dual_coef_)
            model_d.x_support = \
                <double*><uintptr_t>get_dev_array_ptr(self.support_vectors_)
            model_d.support_idx = \
                <int*><uintptr_t>get_dev_array_ptr(self.support_)
            model_d.n_classes = self._n_classes
            if self._n_classes > 0:
                model_d.unique_labels = \
                    <double*><uintptr_t>get_dev_array_ptr(self._unique_labels)
            else:
                model_d.unique_labels = NULL
            return <uintptr_t>model_d

    def _unpack_model(self):
        """ Expose the model parameters as attributes """
        cdef svmModel[float] *model_f
        cdef svmModel[double] *model_d

        # Mark that the C++ layer should free the parameter vectors
        # If we could pass the deviceArray deallocator as finalizer for the
        # device_array_from_ptr function, then this would not be necessary.
        self._freeSvmBuffers = True

        if self.dtype == np.float32:
            model_f = <svmModel[float]*><uintptr_t> self._model
            if model_f.n_support == 0:
                self.fit_status_ = 1  # incorrect fit
                return
            self.intercept_ = model_f.b
            self.n_support_ = model_f.n_support
            self.dual_coef_ = device_array_from_ptr(
                <uintptr_t>model_f.dual_coefs, (1, self.n_support_),
                self.dtype)
            self.support_ = device_array_from_ptr(
                <uintptr_t>model_f.support_idx, (self.n_support_,), np.int32)
            self.support_vectors_ = device_array_from_ptr(
                <uintptr_t>model_f.x_support, (self.n_support_, self.n_cols),
                self.dtype)
            self._n_classes = model_f.n_classes
            if self._n_classes > 0:
                self._unique_labels = device_array_from_ptr(
                    <uintptr_t>model_f.unique_labels, (self._n_classes,),
                    self.dtype)
            else:
                self._unique_labels = None
        else:
            model_d = <svmModel[double]*><uintptr_t> self._model
            if model_d.n_support == 0:
                self.fit_status_ = 1  # incorrect fit
                return
            self.intercept_ = model_d.b
            self.n_support_ = model_d.n_support
            self.dual_coef_ = device_array_from_ptr(
                <uintptr_t>model_d.dual_coefs, (1, self.n_support_),
                self.dtype)
            self.support_ = device_array_from_ptr(
                <uintptr_t>model_d.support_idx, (self.n_support_,), np.int32)
            self.support_vectors_ = device_array_from_ptr(
                <uintptr_t>model_d.x_support, (self.n_support_, self.n_cols),
                self.dtype)
            self._n_classes = model_d.n_classes
            if self._n_classes > 0:
                self._unique_labels = device_array_from_ptr(
                    <uintptr_t>model_d.unique_labels, (self._n_classes,),
                    self.dtype)
            else:
                self._unique_labels = None

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
            svcFit(handle_[0], <float*>X_ptr, <int>self.n_rows,
                   <int>self.n_cols, <float*>y_ptr, param, _kernel_params,
                   model_f[0])
            self._model = <uintptr_t>model_f
        elif self.dtype == np.float64:
            model_d = new svmModel[double]()
            svcFit(handle_[0], <double*>X_ptr, <int>self.n_rows,
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

    def predict(self, X, predict_class):
        """
        Predicts the y for X, where y is either the decision function value
        (if predict_class == False), or the label associated with X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        predict_class : boolean
            Switch whether to retun class label (true), or decision function
            value (false).

        Returns
        -------
        y : cuDF Series
           Dense vector (floats or doubles) of shape (n_samples, 1)
        """

        if self._model is None:
            raise RuntimeError("Call fit before prediction")

        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, pred_dtype = \
            input_to_dev_array(X, check_dtype=self.dtype)

        preds = cudf.Series(zeros(n_rows, dtype=self.dtype))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef svmModel[float]* model_f
        cdef svmModel[double]* model_d

        if self.dtype == np.float32:
            model_f = <svmModel[float]*><size_t> self._model
            svcPredict(handle_[0], <float*>X_ptr, <int>n_rows, <int>n_cols,
                       self._get_kernel_params(), model_f[0],
                       <float*>preds_ptr, <float>self.cache_size,
                       <bool> predict_class)
        else:
            model_d = <svmModel[double]*><size_t> self._model
            svcPredict(handle_[0], <double*>X_ptr, <int>n_rows, <int>n_cols,
                       self._get_kernel_params(), model_d[0],
                       <double*>preds_ptr, <double>self.cache_size,
                       <bool> predict_class)

        self.handle.sync()

        del(X_m)

        return preds

    def get_param_names(self):
        return ["C", "kernel", "degree", "gamma", "coef0", "cache_size",
                "max_iter", "tol", "verbose"]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['handle']
        del state['_model']
        state['dual_coef_'] = cudf.DataFrame.from_gpu_matrix(self.dual_coef_)
        state['support_'] = cudf.Series(self.support_)
        state['support_vectors_'] = \
            cudf.DataFrame.from_gpu_matrix(self.support_vectors_)
        state['_unique_labels'] = cudf.Series(self._unique_labels)

        return state

    def __setstate__(self, state):
        super(SVMBase, self).__init__(handle=None, verbose=state['verbose'])

        state['dual_coef_'] = state['dual_coef_'].as_gpu_matrix()
        state['support_'] = state['support_'].to_gpu_array()
        state['support_vectors_'] = state['support_vectors_'].as_gpu_matrix()
        state['_unique_labels'] = state['_unique_labels'].to_gpu_array()
        self.__dict__.update(state)
        self._model = self._get_svm_model()
        self._freeSvmBuffers = False
