# Copyright (c) 2021, NVIDIA CORPORATION.
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

import re
import typing
import numpy as np
import cupy as cp
import cuml

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.device_uvector cimport device_uvector
cimport rmm._lib.lib as rmm

import cuml.internals
from cython.operator cimport dereference as deref
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin, RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t, _Stream
from cuml.common import input_to_cuml_array, input_to_host_array
from libc.stdint cimport uintptr_t
from libcpp cimport bool as cppbool, nullptr


__all__ = ['LinearSVC', 'LinearSVR']

cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM":

    cdef enum Penalty "ML::SVM::LinearSVMParams::Penalty":
        L1 "ML::SVM::LinearSVMParams::L1"
        L2 "ML::SVM::LinearSVMParams::L2"

    cdef enum Loss "ML::SVM::LinearSVMParams::Loss":
        HINGE "ML::SVM::LinearSVMParams::\
            HINGE"
        SQUARED_HINGE "ML::SVM::LinearSVMParams::\
            SQUARED_HINGE"
        EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
            EPSILON_INSENSITIVE"
        SQUARED_EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
            SQUARED_EPSILON_INSENSITIVE"

    cdef struct LinearSVMParams:
        Penalty penalty
        Loss loss
        cppbool fit_intercept
        cppbool penalized_intercept
        cppbool probability
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        int verbose
        double C
        double grad_tol
        double change_tol
        double svr_sensitivity
        double H1_value

    cdef cppclass LinearSVMModel[T]:
        const handle_t& handle
        device_uvector[T] classes
        device_uvector[T] w
        device_uvector[T] probScale
        LinearSVMModel(
            const handle_t& handle,
            const LinearSVMParams params) except +
        LinearSVMModel(
            const handle_t& handle,
            const LinearSVMParams params,
            const T* X,
            const int nRows,
            const int nCols,
            const T* y,
            const T* sampleWeight) except +
        void predict(
            const T* X, const int nRows, const int nCols, T* out) except +
        void predict_proba(
            const T* X, const int nRows, const int nCols,
            const cppbool log, T* out) except +


cdef class LSVMPWrapper_:
    cdef readonly dict params

    def __cinit__(self, **kwargs):
        cdef LinearSVMParams ps
        self.params = ps

    def _getparam(self, key):
        return self.params[key]

    def _setparam(self, key, val):
        self.params[key] = val

    def __init__(self, **kwargs):
        allowed_keys = set(self.get_param_names())
        for key, val in kwargs.items():
            if key in allowed_keys:
                setattr(self, key, val)

    def get_param_names(self):
        cdef LinearSVMParams ps
        return ps.keys()


# Here we can do custom conversion for selected properties.
class LSVMPWrapper(LSVMPWrapper_):

    @property
    def penalty(self) -> str:
        if self._getparam('penalty') == Penalty.L1:
            return "l1"
        if self._getparam('penalty') == Penalty.L2:
            return "l2"
        raise ValueError(
            f"Unknown penalty enum value: {self._getparam('penalty')}")

    @penalty.setter
    def penalty(self, penalty: str):
        if penalty == "l1":
            self._setparam('penalty', Penalty.L1)
        elif penalty == "l2":
            self._setparam('penalty', Penalty.L2)
        else:
            raise ValueError(f"Unknown penalty string value: {penalty}")

    @property
    def loss(self) -> str:
        loss = self._getparam('loss')
        if loss == Loss.HINGE:
            return "hinge"
        if loss == Loss.SQUARED_HINGE:
            return "squared_hinge"
        if loss == Loss.EPSILON_INSENSITIVE:
            return "epsilon_insensitive"
        if loss == Loss.SQUARED_EPSILON_INSENSITIVE:
            return "squared_epsilon_insensitive"
        raise ValueError(f"Unknown loss enum value: {loss}")

    @loss.setter
    def loss(self, loss: str):
        if loss == "hinge":
            self._setparam('loss', Loss.HINGE)
        elif loss == "squared_hinge":
            self._setparam('loss', Loss.SQUARED_HINGE)
        elif loss == "epsilon_insensitive":
            self._setparam('loss', Loss.EPSILON_INSENSITIVE)
        elif loss == "squared_epsilon_insensitive":
            self._setparam('loss', Loss.SQUARED_EPSILON_INSENSITIVE)
        else:
            raise ValueError(f"Unknown loss string value: {loss}")


# Add properties for parameters with a trivial conversion
def __add_prop(prop_name):
    setattr(LSVMPWrapper, prop_name, property(
        lambda self: self._getparam(prop_name),
        lambda self, value: self._setparam(prop_name, value)
    ))


for prop_name in LSVMPWrapper().get_param_names():
    if not hasattr(LSVMPWrapper, prop_name):
        __add_prop(prop_name)
del __add_prop

LinearSVM_defaults = LSVMPWrapper()
'''Default parameter values for LinearSVM, re-exported from C++.'''

cdef union LinearSVMModelPtr:
    uintptr_t untyped
    LinearSVMModel[float] * float32
    LinearSVMModel[double] * float64

cdef class LinearSVMWrapper:
    cdef readonly object dtype
    cdef object handle
    cdef LinearSVMModelPtr model

    cdef object __coef_
    cdef object __intercept_
    cdef object __classes_
    cdef object __probScale_

    def copy_array(
            self,
            target: CumlArray, source: CumlArray,
            synchronize: bool = True):
        cdef _Stream stream = (
            <handle_t*><size_t>self.handle.getHandle()
            ).get_stream()
        if source.shape != target.shape:
            raise AttributeError(
                f"Expected an array of shape {target.shape}, "
                f"but got {source.shape}")
        if source.dtype != target.dtype:
            raise AttributeError(
                f"Expected an array of type {target.dtype}, "
                f"but got {source.dtype}")
        rmm.cudaMemcpyAsync(
            <void*><uintptr_t>target.ptr,
            <void*><uintptr_t>source.ptr,
            <size_t>(source.nbytes),
            rmm.cudaMemcpyDeviceToDevice,
            stream)
        if synchronize:
            rmm.cudaStreamSynchronize(stream)

    def __cinit__(
            self,
            handle: cuml.Handle,
            paramsWrapper: LSVMPWrapper,
            coefs: typing.Optional[CumlArray] = None,
            intercept: typing.Optional[CumlArray] = None,
            classes: typing.Optional[CumlArray] = None,
            probScale: typing.Optional[CumlArray] = None,
            X: typing.Optional[CumlArray] = None,
            y: typing.Optional[CumlArray] = None,
            sampleWeight: typing.Optional[CumlArray] = None):
        cdef LinearSVMParams params = paramsWrapper.params
        self.handle = handle

        # check if parameters are passed correctly
        do_training = False
        if coefs is None:
            do_training = True
            if X is None or y is None:
                raise TypeError(
                    "You must provide either the weights "
                    "or input data (X, y) to the LinearSVMWrapper")
        else:
            do_training = False
            if coefs.shape[0] > 1 and classes is None:
                raise TypeError(
                    "You must provide classes along with the weights "
                    "to the LinearSVMWrapper classifier")
            if params.probability and probScale is None:
                raise TypeError(
                    "You must provide probability scales "
                    "to the LinearSVMWrapper probabolistic classifier")
            if params.fit_intercept and intercept is None:
                raise TypeError(
                    "You must provide intercept value to the LinearSVMWrapper"
                    " estimator with fit_intercept enabled")

        self.dtype = X.dtype if do_training else coefs.dtype
        cdef handle_t* h = <handle_t*><size_t>handle.getHandle()
        cdef _Stream stream = h.get_stream()
        cdef cuda_stream_view sview = cuda_stream_view(stream)
        nClasses = 0
        nCols = 0
        wCols = 0
        wRows = 0

        if do_training:
            nCols = X.shape[1]
            sw_ptr = sampleWeight.ptr if sampleWeight is not None else 0
            if self.dtype == np.float32:
                self.model.float32 = new LinearSVMModel[float](
                    deref(h), params,
                    <const float*><uintptr_t>X.ptr,
                    X.shape[0], nCols,
                    <const float*><uintptr_t>y.ptr,
                    <const float*><uintptr_t>sw_ptr)
                nClasses = self.model.float32.classes.size()
            elif self.dtype == np.float64:
                self.model.float64 = new LinearSVMModel[double](
                    deref(h), params,
                    <const double*><uintptr_t>X.ptr,
                    X.shape[0], nCols,
                    <const double*><uintptr_t>y.ptr,
                    <const double*><uintptr_t>sw_ptr)
                nClasses = self.model.float64.classes.size()
            else:
                raise TypeError('Input data type should be float32 or float64')
            wCols = 1 if nClasses <= 2 else nClasses
            wRows = nCols + (1 if params.fit_intercept else 0)
        else:
            nCols = coefs.shape[1]
            wCols = coefs.shape[0]
            wRows = nCols + (1 if params.fit_intercept else 0)
            nClasses = classes.shape[0] if classes is not None else 0

            wSize = wCols * wRows
            if self.dtype == np.float32:
                self.model.float32 = new LinearSVMModel[float](
                    deref(h), params)
                self.model.float32.w.resize(wSize, sview)
                if classes is not None:
                    self.model.float32.classes.resize(nClasses, sview)
                if probScale is not None:
                    pSize = probScale.nbytes / self.dtype.itemsize
                    self.model.float32.probScale.resize(pSize, sview)

            elif self.dtype == np.float64:
                self.model.float64 = new LinearSVMModel[double](
                    deref(h), params)
                self.model.float64.w.resize(wSize, sview)
                if classes is not None:
                    self.model.float64.classes.resize(nClasses, sview)
                if probScale is not None:
                    pSize = probScale.nbytes / self.dtype.itemsize
                    self.model.float64.probScale.resize(pSize, sview)
            else:
                raise TypeError('Input data type should be float32 or float64')

        # prepare the attribute arrays
        cdef uintptr_t coef_ptr = 0
        cdef uintptr_t intercept_ptr = 0
        cdef uintptr_t classes_ptr = 0
        cdef uintptr_t probScale_ptr = 0
        if self.dtype == np.float32:
            coef_ptr = <uintptr_t>self.model.float32.w.data()
            intercept_ptr = <uintptr_t>(
                self.model.float32.w.data() + <int>(nCols * wCols))
            classes_ptr = <uintptr_t>self.model.float32.classes.data()
            probScale_ptr = <uintptr_t>self.model.float32.probScale.data()
        elif self.dtype == np.float64:
            coef_ptr = <uintptr_t>self.model.float64.w.data()
            intercept_ptr = <uintptr_t>(
                self.model.float64.w.data() + <int>(nCols * wCols))
            classes_ptr = <uintptr_t>self.model.float64.classes.data()
            probScale_ptr = <uintptr_t>self.model.float64.probScale.data()

        self.__coef_ = CumlArray(
            coef_ptr, dtype=self.dtype,
            shape=(wCols, nCols), owner=self, order='F')
        self.__intercept_ = CumlArray(
            intercept_ptr, dtype=self.dtype,
            shape=(wCols, ), owner=self, order='F'
            ) if params.fit_intercept else None
        self.__classes_ = CumlArray(
            classes_ptr, dtype=self.dtype,
            shape=(nClasses, ), owner=self, order='F'
            ) if nClasses > 0 else None
        self.__probScale_ = CumlArray(
            probScale_ptr, dtype=self.dtype,
            shape=(2 * wCols, ), owner=self, order='F'
            ) if params.probability else None

        # copy the passed state
        if not do_training:
            self.copy_array(self.__coef_, coefs, False)
            if intercept is not None:
                self.copy_array(self.__intercept_, intercept, False)
            if classes is not None:
                self.copy_array(self.__classes_, classes, False)
            if probScale is not None:
                self.copy_array(self.__probScale_, probScale, False)

        handle.sync()

    def __dealloc__(self):
        if self.model.untyped != 0:
            if self.dtype == np.float32:
                del self.model.float32
            elif self.dtype == np.float64:
                del self.model.float64
        self.model.untyped = 0

    @property
    def coef_(self) -> CumlArray:
        return self.__coef_

    @coef_.setter
    def coef_(self, coef: CumlArray):
        self.copy_array(self.__coef_, coef)

    @property
    def intercept_(self) -> CumlArray:
        return self.__intercept_

    @intercept_.setter
    def intercept_(self, intercept: CumlArray):
        self.copy_array(self.__intercept_, intercept)

    @property
    def classes_(self) -> CumlArray:
        return self.__classes_

    @classes_.setter
    def classes_(self, classes: CumlArray):
        self.copy_array(self.__classes_, classes)

    @property
    def probScale_(self) -> CumlArray:
        return self.__probScale_

    @probScale_.setter
    def probScale_(self, probScale: CumlArray):
        self.copy_array(self.__probScale_, probScale)

    def predict(self, X: CumlArray) -> CumlArray:
        y = CumlArray.empty(
            shape=(X.shape[0],),
            dtype=self.dtype, order='C')

        if self.dtype == np.float32:
            self.model.float32.predict(
                <const float*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <float*><uintptr_t>y.ptr)
        elif self.dtype == np.float64:
            self.model.float64.predict(
                <const double*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <double*><uintptr_t>y.ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y

    def predict_proba(self, X, log=False) -> CumlArray:
        y = CumlArray.empty(
            shape=(X.shape[0], self.classes_.shape[0]),
            dtype=self.dtype, order='C')

        if self.dtype == np.float32:
            self.model.float32.predict_proba(
                <const float*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1], log,
                <float*><uintptr_t>y.ptr)
        elif self.dtype == np.float64:
            self.model.float64.predict_proba(
                <const double*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1], log,
                <double*><uintptr_t>y.ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y


class LinearSVM(Base):

    _model_: typing.Optional[LinearSVMWrapper]

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()
    classes_ = CumlArrayDescriptor()
    probScale_ = CumlArrayDescriptor()

    @property
    def model_(self) -> LinearSVMWrapper:
        if self._model_ is None:
            raise AttributeError(
                'The model is not trained yet (call fit() first).')
        return self._model_

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model_'] = None
        return state

    def __init__(self, **kwargs):
        # `tol` is special in that it's not present in get_parameter_names,
        # so having a special logic here does not affect pickling/cloning.
        tol = kwargs.pop('tol', None)
        if tol is not None:
            default_to_ratio = \
                LinearSVM_defaults.change_tol / LinearSVM_defaults.grad_tol
            self.grad_tol = tol
            self.change_tol = tol * default_to_ratio
        # All arguments are optional (they have defaults),
        # yet we need to check for unused arguments
        allowed_keys = set(self.get_param_names())
        super_keys = getattr(super(), 'get_param_names', lambda: [])()
        remaining_kwargs = {}
        for key, val in kwargs.items():
            if key not in allowed_keys or key in super_keys:
                remaining_kwargs[key] = val
                continue
            if val is None:
                continue
            allowed_keys.remove(key)
            setattr(self, key, val)

        # set defaults
        for key in allowed_keys:
            setattr(self, key, getattr(LinearSVM_defaults, key, None))

        super().__init__(**remaining_kwargs)
        self._model_ = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.probScale_ = None

    @property
    @cuml.internals.api_base_return_array_skipall
    def n_classes_(self) -> int:
        if self.classes_ is not None:
            return self.classes_.shape[0]
        return self.model_.classes_.shape[0]

    def get_param_names(self):
        return LinearSVM_defaults.get_param_names()

    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> 'LinearSVM':

        X_m, n_rows, n_cols, self.dtype = input_to_cuml_array(X, order='F')
        convert_to_dtype = self.dtype if convert_dtype else None
        y_m = input_to_cuml_array(
            y, check_dtype=self.dtype,
            convert_to_dtype=convert_to_dtype,
            check_rows=n_rows, check_cols=1).array
        sample_weight_m = input_to_cuml_array(
            sample_weight, check_dtype=self.dtype,
            convert_to_dtype=convert_to_dtype,
            check_rows=n_rows, check_cols=1
            ).array if sample_weight is not None else None

        self._model_ = LinearSVMWrapper(
            handle=self.handle,
            paramsWrapper=LSVMPWrapper(
                **{k: getattr(self, k) for k in self.get_param_names()}
            ),
            X=X_m, y=y_m,
            sampleWeight=sample_weight_m)
        self.coef_ = self._model_.coef_
        self.intercept_ = self._model_.intercept_
        self.classes_ = self._model_.classes_
        self.probScale_ = self._model_.probScale_

        return self

    def __sync_model(self):
        '''
        Update the model on C++ side lazily before calling predict.
        '''
        if self._model_ is None:
            self._model_ = LinearSVMWrapper(
                handle=self.handle,
                paramsWrapper=LSVMPWrapper(
                    **{k: getattr(self, k) for k in self.get_param_names()}
                ),
                coefs=self.coef_,
                intercept=self.intercept_,
                classes=self.classes_,
                probScale=self.probScale_)
            self.coef_ = self._model_.coef_
            self.intercept_ = self._model_.intercept_
            self.classes_ = self._model_.classes_
            self.probScale_ = self._model_.probScale_
        else:
            if self.coef_ is not self.model_.coef_:
                self.model_.coef_ = self.coef_
            if self.intercept_ is not self.model_.intercept_:
                self.model_.intercept_ = self.intercept_
            if self.classes_ is not self.model_.classes_:
                self.model_.classes_ = self.classes_
            if self.probScale_ is not self.model_.probScale_:
                self.model_.probScale_ = self.probScale_

    def predict(self, X, convert_dtype=True) -> CumlArray:
        convert_to_dtype = self.dtype if convert_dtype else None
        X_m, n_rows, n_cols, _ = input_to_cuml_array(
            X, check_dtype=self.dtype,
            convert_to_dtype=convert_to_dtype)
        self.__sync_model()
        return self.model_.predict(X_m)

    def predict_proba(self, X, log=False, convert_dtype=True) -> CumlArray:
        convert_to_dtype = self.dtype if convert_dtype else None
        X_m, n_rows, n_cols, _ = input_to_cuml_array(
            X, check_dtype=self.dtype,
            convert_to_dtype=convert_to_dtype)
        self.__sync_model()
        return self.model_.predict_proba(X_m, log=log)


class LinearSVC(LinearSVM, ClassifierMixin):
    '''
    LinearSVC (Support Vector Classification with the linear kernel)

    Construct a linear SVM classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from cuml.svm import LinearSVC
        X = np.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
                        dtype=np.float32);
        y = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
        clf = LinearSVC(loss='squared_hinge', penalty='l1', C=1)
        clf.fit(X, y)
        print("Predicted labels:", clf.predict(X))

    Output:

    .. code-block:: none

        Predicted labels: [0. 0. 1. 0. 1. 1.]

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    penalty : {{'l1', 'l2'}} (default = '{LinearSVM_defaults.penalty}')
        The regularization term of the target function.
    loss : {LinearSVC.REGISTERED_LOSSES} (default = 'squared_hinge')
        The loss term of the target function.
    fit_intercept : {LinearSVM_defaults.fit_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.fit_intercept})
        Whether to fit the bias term. Set to False if you expect that the
        data is already centered.
    penalized_intercept : { \
            LinearSVM_defaults.penalized_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.penalized_intercept})
        When true, the bias term is treated the same way as other features;
        i.e. it's penalized by the regularization term of the target function.
        Enabling this feature forces an extra copying the input data X.
    max_iter : {LinearSVM_defaults.max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.max_iter})
        Maximum number of iterations for the underlying solver.
    linesearch_max_iter : { \
            LinearSVM_defaults.linesearch_max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.linesearch_max_iter})
        Maximum number of linesearch (inner loop) iterations for
        the underlying (QN) solver.
    lbfgs_memory : { \
            LinearSVM_defaults.lbfgs_memory.__class__.__name__ \
            } (default = {LinearSVM_defaults.lbfgs_memory})
        Number of vectors approximating the hessian for the underlying QN
        solver (l-bfgs).
    verbose : { \
            LinearSVM_defaults.verbose.__class__.__name__ \
            } (default = {LinearSVM_defaults.verbose})
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    C : {LinearSVM_defaults.C.__class__.__name__ \
            } (default = {LinearSVM_defaults.C})
        The constant scaling factor of the loss term in the target formula
          `F(X, y) = penalty(X) + C * loss(X, y)`.
    grad_tol : {LinearSVM_defaults.grad_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.grad_tol})
        The threshold on the gradient for the underlying QN solver.
    change_tol : {LinearSVM_defaults.change_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.change_tol})
        The threshold on the function change for the underlying QN solver.
    tol : Optional[float] (default = None)
        Tolerance for the stopping criterion.
        This is a helper transient parameter that, when present, sets both
        `grad_tol` and `change_tol` to the same value. When any of the two
        `***_tol` parameters are passed as well, they take the precedence.
    H1_value : {LinearSVM_defaults.H1_value.__class__.__name__ \
            } (default = {LinearSVM_defaults.H1_value})
        he value considered 'one' in the binary classification problem.
        This value is converted into `1.0` during training, whereas all the
        other values in the training target data (`y`)
        are converted into `-1.0`.
    probability: bool (default = False)
        Enable or disable probability estimates.
    multiclass_strategy : {{'ovo' or 'ovr'}} (default = 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    output_type : {{'input', 'cudf', 'cupy', 'numpy', 'numba'}} (default=None)
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    intercept_ : float
        The constant in the decision function
    coef_ : float, shape (1, n_cols)
        Only available for linear kernels. It is the normal of the
        hyperplane.
        coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]
    classes_: shape (n_classes_,)
        Array of class labels
    n_classes_ : int
        Number of classes

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVC>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.
    '''

    REGISTERED_LOSSES = set([
        'hinge',
        'squared_hinge'])

    def __init__(self, *args, **kwargs):
        # set classification-specific defaults
        if 'loss' not in kwargs:
            kwargs['loss'] = 'squared_hinge'
        if 'multiclass_strategy' not in kwargs:
            kwargs['multiclass_strategy'] = 'ovr'
        if 'probability' not in kwargs:
            kwargs['probability'] = False

        super().__init__(*args, **kwargs)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Classification loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        self.__loss = loss

    def get_param_names(self):
        return [
            "handle",
            "verbose",
            "output_type",
            'penalty',
            'loss',
            'fit_intercept',
            'penalized_intercept',
            'probability',
            'max_iter',
            'linesearch_max_iter',
            'lbfgs_memory',
            'C',
            'grad_tol',
            'change_tol',
            'H1_value',
            'multiclass_strategy',
        ]


# Format docstring to see the re-exported defaults etc.
LinearSVC.__doc__ = \
    re.sub(r"\{ *([^ ]+) *\}", r"{\1}", LinearSVC.__doc__).format(**locals())


class LinearSVR(LinearSVM, RegressorMixin):
    '''LinearSVR'''

    REGISTERED_LOSSES = set([
        'epsilon_insensitive',
        'squared_epsilon_insensitive'])

    def __init__(self, *args, **kwargs):
        # set regression-specific defaults
        if 'loss' not in kwargs:
            kwargs['loss'] = 'epsilon_insensitive'

        super().__init__(*args, **kwargs)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Regression loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        self.__loss = loss

    def get_param_names(self):
        return [
            "handle",
            "verbose",
            "output_type",
            'penalty',
            'loss',
            'fit_intercept',
            'penalized_intercept',
            'max_iter',
            'linesearch_max_iter',
            'lbfgs_memory',
            'C',
            'grad_tol',
            'change_tol',
            'svr_sensitivity'
        ]


# Format docstring to see the re-exported defaults etc.
LinearSVR.__doc__ = \
    re.sub(r"\{ *([^ ]+) *\}", r"{\1}", LinearSVR.__doc__).format(**locals())


# [WIP]
# !! Checkout requirements in python/cuml/common/base.pyx
#     I must satisfy them to integrate into the cuml ML algo infrastructure
#
# NB: [FEA] scikit-learn based meta estimators
#     https://github.com/rapidsai/cuml/issues/2876
#
#     [REVIEW] Sklearn-based preprocessin
#     https://github.com/rapidsai/cuml/pull/2645
#
#     [FEA] Implement probability calibration using device arrays
#     https://github.com/rapidsai/cuml/issues/2608
#
# Need:
#   1. classes a.k.a. unique_labels
#   2. multiclass strategy
#   3. decide where transform labels into multiclass 1, -1
#   4. Also food for thought:
#
# https://scikit-learn.org/stable/modules/svm.html#svm-multi-class
# https://github.com/scikit-learn/scikit-learn/blob/ccf8e749bac06aa456d8de2d06dbe0d8c507ac3f/sklearn/multiclass.py#L598
