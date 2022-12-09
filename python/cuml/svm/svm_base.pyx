# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# distutils: language = c++

import ctypes
import cupy
import numpy as np

from numba import cuda

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.base import Base
from cuml.common.exceptions import NotFittedError
from pylibraft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common import using_output_type
from cuml.internals.logger import warn
from cuml.internals.mixins import FMajorInputTagMixin
from libcpp cimport bool


cdef extern from "raft/distance/distance_types.hpp" \
        namespace "raft::distance::kernels":
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

    cdef struct SvmParameter:
        # parameters for trainig
        double C
        double cache_size
        int max_iter
        int nochange_steps
        double tol
        int verbosity
        double epsilon
        SvmType svmType

cdef extern from "cuml/svm/svm_model.h" namespace "ML::SVM":
    cdef cppclass SvmModel[math_t]:
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

    cdef void svcPredict[math_t](
        const handle_t &handle, math_t *input, int n_rows, int n_cols,
        KernelParams &kernel_params, const SvmModel[math_t] &model,
        math_t *preds, math_t buffer_size, bool predict_class) except +

    cdef void svmFreeBuffers[math_t](const handle_t &handle,
                                     SvmModel[math_t] &m) except +


class SVMBase(Base,
              FMajorInputTagMixin):
    """
    Base class for Support Vector Machines

    Currently only binary classification is supported.

    The solver uses the SMO method to fit the classifier. We use the Optimized
    Hierarchical Decomposition [1]_ variant of the SMO algorithm, similar to
    [2]_

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
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

        - 'auto': gamma will be set to ``1 / n_features``
        - 'scale': gamma will be se to ``1 / (n_features * X.var())``

    coef0 : float (default = 0.0)
        Independent term in kernel function, only signifficant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
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
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    epsilon: float (default = 0.1)
        epsilon parameter of the epsiron-SVR model. There is no penalty
        associated to points that are predicted within the epsilon-tube
        around the target values.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

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
    intercept_ : float
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    coef_ : float, shape [1, n_cols]
        Only available for linear kernels. It is the normal of the
        hyperplane.
        ``coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]``

    Notes
    -----
    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    References
    ----------
    .. [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical
        Decomposition Algorithm for Support VectorMachine Training, IEEE
        Transactions on Parallel and Distributed Systems, vol 28, no 12, 3330,
        (2017)
    .. [2] `Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs,
        Journal of Machine Learning Research, 19, 1-5 (2018)
        <https://github.com/Xtra-Computing/thundersvm>`_

    """

    dual_coef_ = CumlArrayDescriptor()
    support_ = CumlArrayDescriptor()
    support_vectors_ = CumlArrayDescriptor()
    _intercept_ = CumlArrayDescriptor()
    _internal_coef_ = CumlArrayDescriptor()
    _unique_labels_ = CumlArrayDescriptor()

    def __init__(self, *, handle=None, C=1, kernel='rbf', degree=3,
                 gamma='auto', coef0=0.0, tol=1e-3, cache_size=1024.0,
                 max_iter=-1, nochange_steps=1000, verbose=False,
                 epsilon=0.1, output_type=None):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
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
        self.epsilon = epsilon
        self.svmType = None  # Child class should set self.svmType

        # Parameter to indicate if model has been correctly fitted
        # fit_status == -1 indicates that the model is not yet fitted
        self._fit_status_ = -1

        # Attributes (parameters of the fitted model)
        self.dual_coef_ = None
        self.support_ = None
        self.support_vectors_ = None
        self._intercept_ = None
        self.n_support_ = None

        self._c_kernel = self._get_c_kernel(kernel)
        self._gamma_val = None  # the actual numerical value used for training
        self.coef_ = None  # value of the coef_ attribute, only for lin kernel
        self.dtype = None
        self._model = None  # structure of the model parameters
        self._freeSvmBuffers = False  # whether to call the C++ lib for cleanup

        if (kernel == 'linear' or (kernel == 'poly' and degree == 1)) \
           and not getattr(type(self), "_linear_kernel_warned", False):
            setattr(type(self), "_linear_kernel_warned", True)
            cname = type(self).__name__
            warn(f'{cname} with the linear kernel can be much faster using '
                 f'the specialized solver provided by Linear{cname}. Consider '
                 f'switching to Linear{cname} if tranining takes too long.')

    def __del__(self):
        self._dealloc()

    def _dealloc(self):
        # deallocate model parameters
        cdef SvmModel[float] *model_f
        cdef SvmModel[double] *model_d
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if self._model is not None:
            if self.dtype == np.float32:
                model_f = <SvmModel[float]*><uintptr_t> self._model
                if self._freeSvmBuffers:
                    svmFreeBuffers(handle_[0], model_f[0])
                del model_f
            elif self.dtype == np.float64:
                model_d = <SvmModel[double]*><uintptr_t> self._model
                if self._freeSvmBuffers:
                    svmFreeBuffers(handle_[0], model_d[0])
                del model_d
            else:
                raise TypeError("Unknown type for SVC class")
            try:
                del self._fit_status_
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
                x_var = cupy.asarray(X).var().item()
                return 1 / (self.n_cols * x_var)
            else:
                raise ValueError("Not implemented gamma option: " + self.gamma)
        else:
            return self.gamma

    def _calc_coef(self):
        if (self.n_support_ == 0):
            return cupy.zeros((1, self.n_cols), dtype=self.dtype)
        with using_output_type("cupy"):
            return cupy.dot(self.dual_coef_, self.support_vectors_)

    def _check_is_fitted(self, attr):
        if not hasattr(self, attr) or (getattr(self, attr) is None):
            msg = ("This classifier instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)

    @property
    @cuml.internals.api_base_return_array_skipall
    def coef_(self):
        if self._c_kernel != LINEAR:
            raise AttributeError("coef_ is only available for linear kernels")
        if self._model is None:
            raise AttributeError("Call fit before prediction")
        if self._internal_coef_ is None:
            self._internal_coef_ = self._calc_coef()
        # Call the base class to perform the output conversion
        return self._internal_coef_

    @coef_.setter
    def coef_(self, value):
        self._internal_coef_ = value

    @property
    @cuml.internals.api_base_return_array_skipall
    def intercept_(self):
        if self._intercept_ is None:
            raise AttributeError("intercept_ called before fit.")
        return self._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

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
        """ Wrap the training parameters in an SvmParameter obtect """
        cdef SvmParameter param
        param.C = self.C
        param.cache_size = self.cache_size
        param.max_iter = self.max_iter
        param.nochange_steps = self.nochange_steps
        param.tol = self.tol
        param.verbosity = self.verbose
        param.epsilon = self.epsilon
        param.svmType = self.svmType
        return param

    @cuml.internals.api_base_return_any_skipall
    def _get_svm_model(self):
        """ Wrap the fitted model parameters into an SvmModel structure.
        This is used if the model is loaded by pickle, the self._model struct
        that we can pass to the predictor.
        """
        cdef SvmModel[float] *model_f
        cdef SvmModel[double] *model_d
        if self.dual_coef_ is None:
            # the model is not fitted in this case
            return None
        if self.dtype == np.float32:
            model_f = new SvmModel[float]()
            model_f.n_support = self.n_support_
            model_f.n_cols = self.n_cols
            model_f.b = self._intercept_.item()
            model_f.dual_coefs = \
                <float*><size_t>self.dual_coef_.ptr
            model_f.x_support = \
                <float*><uintptr_t>self.support_vectors_.ptr
            model_f.support_idx = \
                <int*><uintptr_t>self.support_.ptr
            model_f.n_classes = self.n_classes_
            if self.n_classes_ > 0:
                model_f.unique_labels = \
                    <float*><uintptr_t>self._unique_labels_.ptr
            else:
                model_f.unique_labels = NULL
            return <uintptr_t>model_f
        else:
            model_d = new SvmModel[double]()
            model_d.n_support = self.n_support_
            model_d.n_cols = self.n_cols
            model_d.b = self._intercept_.item()
            model_d.dual_coefs = \
                <double*><size_t>self.dual_coef_.ptr
            model_d.x_support = \
                <double*><uintptr_t>self.support_vectors_.ptr
            model_d.support_idx = \
                <int*><uintptr_t>self.support_.ptr
            model_d.n_classes = self.n_classes_
            if self.n_classes_ > 0:
                model_d.unique_labels = \
                    <double*><uintptr_t>self._unique_labels_.ptr
            else:
                model_d.unique_labels = NULL
            return <uintptr_t>model_d

    def _unpack_model(self):
        """ Expose the model parameters as attributes """
        cdef SvmModel[float] *model_f
        cdef SvmModel[double] *model_d

        # Mark that the C++ layer should free the parameter vectors
        # If we could pass the deviceArray deallocator as finalizer for the
        # device_array_from_ptr function, then this would not be necessary.
        self._freeSvmBuffers = True

        if self.dtype == np.float32:
            model_f = <SvmModel[float]*><uintptr_t> self._model
            self._intercept_ = CumlArray.full(1, model_f.b, np.float32)
            self.n_support_ = model_f.n_support

            if model_f.n_support > 0:
                self.dual_coef_ = CumlArray(
                    data=<uintptr_t>model_f.dual_coefs,
                    shape=(1, self.n_support_),
                    dtype=self.dtype,
                    order='F')

                self.support_ = CumlArray(
                    data=<uintptr_t>model_f.support_idx,
                    shape=(self.n_support_,),
                    dtype=np.int32,
                    order='F')

                self.support_vectors_ = CumlArray(
                    data=<uintptr_t>model_f.x_support,
                    shape=(self.n_support_, self.n_cols),
                    dtype=self.dtype,
                    order='F')

            self.n_classes_ = model_f.n_classes
            if self.n_classes_ > 0:
                self._unique_labels_ = CumlArray(
                    data=<uintptr_t>model_f.unique_labels,
                    shape=(self.n_classes_,),
                    dtype=self.dtype,
                    order='F')
            else:
                self._unique_labels_ = None
        else:
            model_d = <SvmModel[double]*><uintptr_t> self._model
            self._intercept_ = CumlArray.full(1, model_d.b, np.float64)
            self.n_support_ = model_d.n_support

            if model_d.n_support > 0:
                self.dual_coef_ = CumlArray(
                    data=<uintptr_t>model_d.dual_coefs,
                    shape=(1, self.n_support_),
                    dtype=self.dtype,
                    order='F')

                self.support_ = CumlArray(
                    data=<uintptr_t>model_d.support_idx,
                    shape=(self.n_support_,),
                    dtype=np.int32,
                    order='F')

                self.support_vectors_ = CumlArray(
                    data=<uintptr_t>model_d.x_support,
                    shape=(self.n_support_, self.n_cols),
                    dtype=self.dtype,
                    order='F')

            self.n_classes_ = model_d.n_classes
            if self.n_classes_ > 0:
                self._unique_labels_ = CumlArray(
                    data=<uintptr_t>model_d.unique_labels,
                    shape=(self.n_classes_,),
                    dtype=self.dtype,
                    order='F')
            else:
                self._unique_labels_ = None

        if self.n_support_ == 0:
            self.dual_coef_ = CumlArray.empty(
                shape=(1, 0),
                dtype=self.dtype,
                order='F')

            self.support_ = CumlArray.empty(
                shape=(0,),
                dtype=np.int32,
                order='F')

            # Setting all dims to zero due to issue
            # https://github.com/rapidsai/cuml/issues/4095
            self.support_vectors_ = CumlArray.empty(
                shape=(0, 0),
                dtype=self.dtype,
                order='F')

    def predict(self, X, predict_class, convert_dtype=True) -> CumlArray:
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
        if predict_class:
            out_dtype = self._get_target_dtype()
        else:
            out_dtype = self.dtype

        cuml.internals.set_api_output_dtype(out_dtype)

        self._check_is_fitted('_model')

        X_m, n_rows, n_cols, pred_dtype = \
            input_to_cuml_array(
                X,
                check_dtype=self.dtype,
                convert_to_dtype=(self.dtype if convert_dtype else None))

        cdef uintptr_t X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows, dtype=self.dtype, index=X_m.index)
        cdef uintptr_t preds_ptr = preds.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef SvmModel[float]* model_f
        cdef SvmModel[double]* model_d

        if self.dtype == np.float32:
            model_f = <SvmModel[float]*><size_t> self._model
            svcPredict(handle_[0], <float*>X_ptr, <int>n_rows, <int>n_cols,
                       self._get_kernel_params(), model_f[0],
                       <float*>preds_ptr, <float>self.cache_size,
                       <bool> predict_class)
        else:
            model_d = <SvmModel[double]*><size_t> self._model
            svcPredict(handle_[0], <double*>X_ptr, <int>n_rows, <int>n_cols,
                       self._get_kernel_params(), model_d[0],
                       <double*>preds_ptr, <double>self.cache_size,
                       <bool> predict_class)

        self.handle.sync()

        del(X_m)

        return preds

    def get_param_names(self):
        return super().get_param_names() + [
            "C",
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "tol",
            "cache_size",
            "max_iter",
            "nochange_steps",
            "epsilon",
        ]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['handle']
        del state['_model']
        return state

    def __setstate__(self, state):
        super(SVMBase, self).__init__(handle=None,
                                      verbose=state['verbose'])
        self.__dict__.update(state)
        self._model = self._get_svm_model()
        self._freeSvmBuffers = False
