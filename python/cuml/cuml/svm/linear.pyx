# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import inspect
import re
import typing

import numpy as np

import cuml
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)

from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from collections import OrderedDict

from cython.operator cimport dereference as deref

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.base_helpers import BaseMetaClass
from cuml.internals.mixins import RegressorMixin

from cuml.internals.logger cimport level_enum

from cuml.internals import logger

from pylibraft.common.handle cimport handle_t

from pylibraft.common.interruptible import cuda_interruptible

from cuml.common import input_to_cuml_array

from cuda.bindings.cyruntime cimport cudaMemcpyAsync, cudaMemcpyKind
from libc.stdint cimport uintptr_t
from libcpp cimport bool as cppbool

__all__ = ['LinearSVM', 'LinearSVM_defaults']


cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM" nogil:

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
        level_enum verbose
        double C
        double grad_tol
        double change_tol
        double epsilon

    cdef cppclass LinearSVMModel[T]:
        const handle_t& handle
        T* classes
        T* w
        T* probScale
        size_t nClasses
        size_t coefRows
        size_t coefCols()

        @staticmethod
        LinearSVMModel[T] allocate(
            const handle_t& handle,
            const LinearSVMParams& params,
            const size_t nCols,
            const size_t nClasses) except +

        @staticmethod
        void free(
            const handle_t& handle,
            const LinearSVMModel[T]& model) except +

        @staticmethod
        LinearSVMModel[T] fit(
            const handle_t& handle,
            const LinearSVMParams& params,
            const T* X,
            const size_t nRows,
            const size_t nCols,
            const T* y,
            const T* sampleWeight) except +

        @staticmethod
        void predict(
            const handle_t& handle,
            const LinearSVMParams& params,
            const LinearSVMModel[T]& model,
            const T* X,
            const size_t nRows, const size_t nCols, T* out) except +

        @staticmethod
        void decisionFunction(
            const handle_t& handle,
            const LinearSVMParams& params,
            const LinearSVMModel[T]& model,
            const T* X,
            const size_t nRows, const size_t nCols, T* out) except +

        @staticmethod
        void predictProba(
            const handle_t& handle,
            const LinearSVMParams& params,
            const LinearSVMModel[T]& model,
            const T* X,
            const size_t nRows, const size_t nCols,
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
        allowed_keys = set(self._get_param_names())
        for key, val in kwargs.items():
            if key in allowed_keys:
                setattr(self, key, val)

    @classmethod
    def _get_param_names(cls):
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

    @property
    def verbose(self):
        return logger._verbose_from_level(self._getparams('verbose'))

    @verbose.setter
    def verbose(self, level: int):
        self._setparam('verbose', logger._verbose_to_level(level))


# Add properties for parameters with a trivial conversion
def __add_prop(prop_name):
    setattr(LSVMPWrapper, prop_name, property(
        lambda self: self._getparam(prop_name),
        lambda self, value: self._setparam(prop_name, value)
    ))


for prop_name in LSVMPWrapper()._get_param_names():
    if not hasattr(LSVMPWrapper, prop_name):
        __add_prop(prop_name)
del __add_prop

LinearSVM_defaults = LSVMPWrapper()
# Default parameter values for LinearSVM, re-exported from C++.

cdef union SomeLinearSVMModel:
    LinearSVMModel[float] float32
    LinearSVMModel[double] float64

cdef class LinearSVMWrapper:
    cdef readonly object dtype
    cdef handle_t* handle
    cdef LinearSVMParams params
    cdef SomeLinearSVMModel model

    cdef object __coef_
    cdef object __intercept_
    cdef object __classes_
    cdef object __probScale_

    def copy_array(
            self,
            target: CumlArray, source: CumlArray,
            synchronize: bool = True):
        cdef cuda_stream_view stream = self.handle.get_stream()
        if source.shape != target.shape:
            raise AttributeError(
                f"Expected an array of shape {target.shape}, "
                f"but got {source.shape}")
        if source.dtype != target.dtype:
            raise AttributeError(
                f"Expected an array of type {target.dtype}, "
                f"but got {source.dtype}")
        cudaMemcpyAsync(
            <void*><uintptr_t>target.ptr,
            <void*><uintptr_t>source.ptr,
            <size_t>(source.size),
            cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            stream.value())
        if synchronize:
            self.handle.sync_stream()

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
        self.handle = <handle_t*><size_t>handle.getHandle()
        self.params = paramsWrapper.params

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
            if self.params.probability and probScale is None:
                raise TypeError(
                    "You must provide probability scales "
                    "to the LinearSVMWrapper probabolistic classifier")
            if self.params.fit_intercept and intercept is None:
                raise TypeError(
                    "You must provide intercept value to the LinearSVMWrapper"
                    " estimator with fit_intercept enabled")

        self.dtype = X.dtype if do_training else coefs.dtype
        nClasses = 0

        if self.dtype != np.float32 and self.dtype != np.float64:
            raise TypeError('Input data type must be float32 or float64')

        cdef uintptr_t Xptr = <uintptr_t>X.ptr if X is not None else 0
        cdef uintptr_t yptr = <uintptr_t>y.ptr if y is not None else 0
        cdef uintptr_t swptr = <uintptr_t>sampleWeight.ptr \
            if sampleWeight is not None else 0
        cdef size_t nCols = 0
        cdef size_t nRows = 0
        if do_training:
            nCols = X.shape[1] if len(X.shape) == 2 else 1
            nRows = X.shape[0]
            if self.dtype == np.float32:
                with cuda_interruptible():
                    with nogil:
                        self.model.float32 = LinearSVMModel[float].fit(
                            deref(self.handle), self.params,
                            <const float*>Xptr,
                            nRows, nCols,
                            <const float*>yptr,
                            <const float*>swptr)
                nClasses = self.model.float32.nClasses
            elif self.dtype == np.float64:
                with cuda_interruptible():
                    with nogil:
                        self.model.float64 = LinearSVMModel[double].fit(
                            deref(self.handle), self.params,
                            <const double*>Xptr,
                            nRows, nCols,
                            <const double*>yptr,
                            <const double*>swptr)
                nClasses = self.model.float64.nClasses
        else:
            nCols = coefs.shape[1]
            nClasses = classes.shape[0] if classes is not None else 0
            if self.dtype == np.float32:
                self.model.float32 = LinearSVMModel[float].allocate(
                    deref(self.handle), self.params, nCols, nClasses)
            elif self.dtype == np.float64:
                self.model.float64 = LinearSVMModel[double].allocate(
                    deref(self.handle), self.params, nCols, nClasses)

        # prepare the attribute arrays
        cdef uintptr_t coef_ptr = 0
        cdef uintptr_t intercept_ptr = 0
        cdef uintptr_t classes_ptr = 0
        cdef uintptr_t probScale_ptr = 0
        wCols = 0
        wRows = 0
        if self.dtype == np.float32:
            wCols = self.model.float32.coefCols()
            wRows = self.model.float32.coefRows
            coef_ptr = <uintptr_t>self.model.float32.w
            intercept_ptr = <uintptr_t>(
                self.model.float32.w + <int>(nCols * wCols))
            classes_ptr = <uintptr_t>self.model.float32.classes
            probScale_ptr = <uintptr_t>self.model.float32.probScale
        elif self.dtype == np.float64:
            wCols = self.model.float64.coefCols()
            wRows = self.model.float64.coefRows
            coef_ptr = <uintptr_t>self.model.float64.w
            intercept_ptr = <uintptr_t>(
                self.model.float64.w + <int>(nCols * wCols))
            classes_ptr = <uintptr_t>self.model.float64.classes
            probScale_ptr = <uintptr_t>self.model.float64.probScale

        self.__coef_ = CumlArray(
            coef_ptr, dtype=self.dtype,
            shape=(wCols, nCols), owner=self, order='F')
        self.__intercept_ = CumlArray(
            intercept_ptr, dtype=self.dtype,
            shape=(wCols, ), owner=self, order='F'
            ) if self.params.fit_intercept else None
        self.__classes_ = CumlArray(
            classes_ptr, dtype=self.dtype,
            shape=(nClasses, ), owner=self, order='F'
            ) if nClasses > 0 else None
        self.__probScale_ = CumlArray(
            probScale_ptr, dtype=self.dtype,
            shape=(wCols, 2), owner=self, order='F'
            ) if self.params.probability else None

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
        if self.dtype == np.float32:
            LinearSVMModel[float].free(
                deref(self.handle), self.model.float32)
        elif self.dtype == np.float64:
            LinearSVMModel[double].free(
                deref(self.handle), self.model.float64)

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
            LinearSVMModel[float].predict(
                deref(self.handle),
                self.params,
                self.model.float32,
                <const float*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <float*><uintptr_t>y.ptr)
        elif self.dtype == np.float64:
            LinearSVMModel[double].predict(
                deref(self.handle),
                self.params,
                self.model.float64,
                <const double*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <double*><uintptr_t>y.ptr)
        else:
            raise TypeError('Input data type must be float32 or float64')

        return y

    def decision_function(self, X: CumlArray) -> CumlArray:
        n_classes = self.classes_.shape[0]
        # special handling of binary case
        shape = (X.shape[0],) if n_classes <= 2 else (X.shape[0], n_classes)
        y = CumlArray.empty(
            shape=shape,
            dtype=self.dtype, order='C')

        if self.dtype == np.float32:
            LinearSVMModel[float].decisionFunction(
                deref(self.handle),
                self.params,
                self.model.float32,
                <const float*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <float*><uintptr_t>y.ptr)
        elif self.dtype == np.float64:
            LinearSVMModel[double].decisionFunction(
                deref(self.handle),
                self.params,
                self.model.float64,
                <const double*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1],
                <double*><uintptr_t>y.ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y

    def predict_proba(self, X, *, log=False) -> CumlArray:
        y = CumlArray.empty(
            shape=(X.shape[0], self.classes_.shape[0]),
            dtype=self.dtype, order='C')

        if self.dtype == np.float32:
            LinearSVMModel[float].predictProba(
                deref(self.handle),
                self.params,
                self.model.float32,
                <const float*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1], log,
                <float*><uintptr_t>y.ptr)
        elif self.dtype == np.float64:
            LinearSVMModel[double].predictProba(
                deref(self.handle),
                self.params,
                self.model.float64,
                <const double*><uintptr_t>X.ptr,
                X.shape[0], X.shape[1], log,
                <double*><uintptr_t>y.ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y


class WithReexportedParams(BaseMetaClass):
    '''Additional post-processing for children of the base class:

        1. Adds keyword arguments from the base classes to the signature.
           Note, this does not affect __init__ method itself, only its
           signature - i.e. how it appears in inspect/help/docs.
           __init__ method must have `**kwargs` argument for this to
           make sense.

        2. Applies string.format() to the class docstring with the globals()
           in the scope of the __init__ method.
           This allows to write variable names (e.g. some constants) in docs,
           such that they are substituted with their actual values.
           Useful for reexporting default values from somewhere else.
    '''

    def __new__(cls, name, bases, attrs):

        def get_class_params(init, parents):
            # collect the keyword arguments from the class hierarchy
            params = OrderedDict()
            for k in parents:
                params.update(
                    get_class_params(getattr(k, '__init__', None), k.__bases__)
                )
            if init is not None:
                sig = inspect.signature(init)
                for k, v in sig.parameters.items():
                    if v.kind == inspect.Parameter.KEYWORD_ONLY:
                        params[k] = v
                del sig
            return params

        init = attrs.get('__init__', None)
        if init is not None:
            # insert keyword arguments from parents
            ppams = get_class_params(init, bases)
            sig = inspect.signature(init)
            params = [
                p for p in sig.parameters.values()
                if p.kind != inspect.Parameter.KEYWORD_ONLY
            ]
            params[1:1] = ppams.values()
            attrs['__init__'].__signature__ = sig.replace(parameters=params)
            del sig

            # format documentation -- replace variables with values
            doc = attrs.get('__doc__', None)
            if doc is not None:
                globs = init.__globals__.copy()
                globs[name] = type(name, (), attrs)
                attrs['__doc__'] = \
                    re.sub(r"\{ *([^ ]+) *\}", r"{\1}", doc).format(**globs)
                del globs
            del doc
        del init
        return super().__new__(cls, name, bases, attrs)


class LinearSVM(Base, InteropMixin, metaclass=WithReexportedParams):

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
        # `tol` is special in that it's not present in _get_param_names,
        # so having a special logic here does not affect pickling/cloning.
        tol = kwargs.pop('tol', None)
        if tol is not None:
            default_to_ratio = \
                LinearSVM_defaults.change_tol / LinearSVM_defaults.grad_tol
            self.grad_tol = tol
            self.change_tol = tol * default_to_ratio
        # All arguments are optional (they have defaults),
        # yet we need to check for unused arguments
        allowed_keys = set(self._get_param_names())
        super_keys = set(super()._get_param_names())
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

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "penalty",
            "loss",
            "fit_intercept",
            "penalized_intercept",
            "probability",
            "max_iter",
            "linesearch_max_iter",
            "lbfgs_memory",
            "C",
            "grad_tol",
            "change_tol",
            "epsilon",
            "tol"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.intercept_scaling != 1:
            raise UnsupportedOnGPU(
                f"`intercept_scaling={model.intercept_scaling}` is not supported"
            )
        params = {
            "C": model.C,
            "fit_intercept": model.fit_intercept,
            "max_iter": model.max_iter,
            "tol": model.tol,
        }

        return params

    def _params_to_cpu(self):
        params = {
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.grad_tol,
        }

        return params

    def _attrs_from_cpu(self, model):
        coef_ = model.coef_.reshape(1, -1)
        return {
            "coef_": to_gpu(coef_, order="F", dtype=np.float64),
            "intercept_": to_gpu(model.intercept_, order="F", dtype=np.float64),
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        coef = self.coef_
        if (coef is not None and coef.ndim == 2 and coef.shape[0] == 1 and
                isinstance(self, RegressorMixin)):
            coef = self.coef_[0]

        return {
            "coef_": to_cpu(coef, order="C", dtype=np.float64),
            "intercept_": to_cpu(self.intercept_, order="C", dtype=np.float64),
            **super()._attrs_to_cpu(model)
        }

    def _sync_attrs_from_cpu(self, model):
        super()._sync_attrs_from_cpu(model)
        self.__sync_model()

    @property
    def n_classes_(self) -> int:
        if self.classes_ is not None:
            return self.classes_.shape[0]
        return self.model_.classes_.shape[0]

    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> 'LinearSVM':
        X_m, n_rows, self.n_features_in_, dtype = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order='F')

        convert_to_dtype = dtype if convert_dtype else None
        y_m = input_to_cuml_array(
            y, check_dtype=dtype,
            convert_to_dtype=convert_to_dtype,
            check_rows=n_rows, check_cols=1).array

        if X.size == 0 or y.size == 0:
            raise ValueError("empty data")

        sample_weight_m = input_to_cuml_array(
            sample_weight, check_dtype=dtype,
            convert_to_dtype=convert_to_dtype,
            check_rows=n_rows, check_cols=1
            ).array if sample_weight is not None else None

        self._model_ = LinearSVMWrapper(
            handle=self.handle,
            paramsWrapper=LSVMPWrapper(**self.get_params()),
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
                paramsWrapper=LSVMPWrapper(**self.get_params()),
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

    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        current_dtype = self.coef_.dtype
        convert_to_dtype = current_dtype if convert_dtype else None
        X_m, _, n_features, _ = input_to_cuml_array(
            X, check_dtype=current_dtype,
            convert_to_dtype=convert_to_dtype)

        if n_features != self.n_features_in_:
            raise ValueError("Reshape your data")

        self.__sync_model()
        return self.model_.predict(X_m)

    def decision_function(self, X, *, convert_dtype=True) -> CumlArray:
        current_dtype = self.coef_.dtype
        convert_to_dtype = current_dtype if convert_dtype else None
        X_m, _, _, _ = input_to_cuml_array(
            X, check_dtype=current_dtype,
            convert_to_dtype=convert_to_dtype)
        self.__sync_model()
        return self.model_.decision_function(X_m)

    def predict_proba(self, X, *, log=False, convert_dtype=True) -> CumlArray:
        current_dtype = self.coef_.dtype
        convert_to_dtype = current_dtype if convert_dtype else None
        X_m, _, _, _ = input_to_cuml_array(
            X, check_dtype=current_dtype,
            convert_to_dtype=convert_to_dtype)
        self.__sync_model()
        return self.model_.predict_proba(X_m, log=log)
