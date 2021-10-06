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

import typing
import numpy as np
import cuml

from cython.operator cimport dereference as deref
from cuml.common.array import CumlArray
from cuml.common.mixins import ClassifierMixin, RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from libc.stdint cimport uintptr_t
from libcpp cimport bool as cppbool, nullptr

cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM":

    cdef cppclass LinearSVMParams:
        enum Penalty:
            L1 "ML::SVM::LinearSVMParams::L1"
            L2 "ML::SVM::LinearSVMParams::L2"

        enum Loss:
            HINGE "ML::SVM::LinearSVMParams::\
                HINGE"
            SQUARED_HINGE "ML::SVM::LinearSVMParams::\
                SQUARED_HINGE"
            EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
                EPSILON_INSENSITIVE"
            SQUARED_EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
                SQUARED_EPSILON_INSENSITIVE"

        Penalty penalty
        Loss loss
        cppbool fit_intercept
        cppbool penalized_intercept
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        int verbosity
        double C
        double grad_tol
        double change_tol
        double svr_sensitivity

    cdef cppclass LinearSVMModel[T]:
        const handle_t& handle
        const int nRows
        const int nCols

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
        T getIntercept()
        T* getCoefsPtr()
        int getCoefsCount()


cdef union LinearSVMModelPtr:
    uintptr_t untyped
    LinearSVMModel[float] * float32
    LinearSVMModel[double] * float64


cdef class LinearSVM:
    cdef LinearSVMParams params
    cdef readonly object handle
    cdef readonly object dtype
    cdef LinearSVMModelPtr model

    non_default_attrs: set

    cdef void reset_model(self):
        if self.model.untyped != 0:
            if self.dtype == np.float32:
                del self.model.float32
            elif self.dtype == np.float64:
                del self.model.float64
        self.model.untyped = 0
        self.dtype = None

    def __cinit__(self, handle: typing.Optional[cuml.Handle] = None, **kwargs):
        self.non_default_attrs = set()
        self.handle = handle if handle is not None else cuml.Handle()
        self.reset_model()

    def __dealloc__(self):
        self.reset_model()

    def __getnewargs_ex__(self):
        return (), {'handle': self.handle}

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.non_default_attrs}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __init__(
            self,
            handle: typing.Optional[cuml.Handle] = None,
            penalty: typing.Optional[str] = None,
            loss: typing.Optional[str] = None,
            tol: typing.Optional[float] = None,
            fit_intercept: typing.Optional[bool] = None,
            penalized_intercept: typing.Optional[bool] = None,
            verbose: typing.Optional[int] = None,
            max_iter: typing.Optional[int] = None,
            linesearch_max_iter: typing.Optional[int] = None,
            lbfgs_memory: typing.Optional[int] = None,
            C: typing.Optional[float] = None,
            grad_tol: typing.Optional[float] = None,
            change_tol: typing.Optional[float] = None,
            svr_sensitivity: typing.Optional[float] = None):
        super().__init__()

        if penalty is not None:
            self.penalty = penalty
        if loss is not None:
            self.loss = loss
        if fit_intercept is not None:
            self.fit_intercept = fit_intercept
        if penalized_intercept is not None:
            self.penalized_intercept = penalized_intercept
        if verbose is not None:
            self.verbosity = verbose
        if max_iter is not None:
            self.max_iter = max_iter
        if linesearch_max_iter is not None:
            self.linesearch_max_iter = linesearch_max_iter
        if lbfgs_memory is not None:
            self.lbfgs_memory = lbfgs_memory
        if C is not None:
            self.C = C
        if svr_sensitivity is not None:
            self.svr_sensitivity = svr_sensitivity

        if grad_tol is not None:
            self.grad_tol = grad_tol
        elif tol is not None:
            self.grad_tol = tol

        if change_tol is not None:
            self.change_tol = change_tol
        elif tol is not None:
            self.change_tol = tol

    @property
    def intercept_(self):
        if self.dtype == np.float32:
            return [self.model.float32.getIntercept()]
        if self.dtype == np.float64:
            return [self.model.float64.getIntercept()]
        raise AttributeError('Train the model first')

    @property
    def coef_(self):
        cdef uintptr_t w_ptr
        k = 0
        if self.dtype == np.float32:
            k = self.model.float32.getCoefsCount()
            w_ptr = <uintptr_t> self.model.float32.getCoefsPtr()
        elif self.dtype == np.float64:
            k = self.model.float64.getCoefsCount()
            w_ptr = <uintptr_t> self.model.float64.getCoefsPtr()
        else:
            raise AttributeError('Train the model first')
        return CumlArray(
            w_ptr, dtype=self.dtype, shape=(1, k), owner=self, order='F'
            ).to_output(output_type='numpy')

    @property
    def penalty(self) -> str:
        if self.params.penalty == LinearSVMParams.Penalty.L1:
            return "l1"
        if self.params.penalty == LinearSVMParams.Penalty.L2:
            return "l2"
        raise AttributeError(
            f"Unknown penalty enum value: {self.params.penalty}")

    @penalty.setter
    def penalty(self, penalty: str):
        if penalty == "l1":
            self.params.penalty = LinearSVMParams.Penalty.L1
        elif penalty == "l2":
            self.params.penalty = LinearSVMParams.Penalty.L2
        else:
            raise AttributeError(f"Unknown penalty string value: {penalty}")
        self.non_default_attrs.add('penalty')

    @property
    def loss(self) -> str:
        loss = self.params.loss
        if loss == LinearSVMParams.Loss.HINGE:
            return "hinge"
        if loss == LinearSVMParams.Loss.SQUARED_HINGE:
            return "squared_hinge"
        if loss == LinearSVMParams.Loss.EPSILON_INSENSITIVE:
            return "epsilon_insensitive"
        if loss == LinearSVMParams.Loss.SQUARED_EPSILON_INSENSITIVE:
            return "squared_epsilon_insensitive"
        raise AttributeError(f"Unknown loss enum value: {loss}")

    @loss.setter
    def loss(self, loss: str):
        if loss == "hinge":
            self.params.loss = LinearSVMParams.Loss.HINGE
        elif loss == "squared_hinge":
            self.params.loss = LinearSVMParams.Loss.SQUARED_HINGE
        elif loss == "epsilon_insensitive":
            self.params.loss = LinearSVMParams.Loss.EPSILON_INSENSITIVE
        elif loss == "squared_epsilon_insensitive":
            self.params.loss = LinearSVMParams.Loss.SQUARED_EPSILON_INSENSITIVE
        else:
            raise AttributeError(f"Unknown loss string value: {loss}")
        self.non_default_attrs.add('loss')

    @property
    def fit_intercept(self) -> bool:
        return self.params.fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, fit_intercept: bool):
        self.params.fit_intercept = fit_intercept
        self.non_default_attrs.add('fit_intercept')

    @property
    def penalized_intercept(self) -> bool:
        return self.params.penalized_intercept

    @penalized_intercept.setter
    def penalized_intercept(self, penalized_intercept: bool):
        self.params.penalized_intercept = penalized_intercept
        self.non_default_attrs.add('penalized_intercept')

    @property
    def max_iter(self) -> int:
        return self.params.max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        self.params.max_iter = max_iter
        self.non_default_attrs.add('max_iter')

    @property
    def linesearch_max_iter(self) -> int:
        return self.params.linesearch_max_iter

    @linesearch_max_iter.setter
    def linesearch_max_iter(self, linesearch_max_iter: int):
        self.params.linesearch_max_iter = linesearch_max_iter
        self.non_default_attrs.add('linesearch_max_iter')

    @property
    def lbfgs_memory(self) -> int:
        return self.params.lbfgs_memory

    @lbfgs_memory.setter
    def lbfgs_memory(self, lbfgs_memory: int):
        self.params.lbfgs_memory = lbfgs_memory
        self.non_default_attrs.add('lbfgs_memory')

    @property
    def verbosity(self) -> int:
        return self.params.verbosity

    @verbosity.setter
    def verbosity(self, verbosity: int):
        self.params.verbosity = verbosity
        self.non_default_attrs.add('verbosity')

    @property
    def C(self) -> float:
        return self.params.C

    @C.setter
    def C(self, C: float):
        self.params.C = C
        self.non_default_attrs.add('C')

    @property
    def grad_tol(self) -> float:
        return self.params.grad_tol

    @grad_tol.setter
    def grad_tol(self, grad_tol: float):
        self.params.grad_tol = grad_tol
        self.non_default_attrs.add('grad_tol')

    @property
    def change_tol(self) -> float:
        return self.params.change_tol

    @change_tol.setter
    def change_tol(self, change_tol: float):
        self.params.change_tol = change_tol
        self.non_default_attrs.add('change_tol')

    @property
    def svr_sensitivity(self) -> float:
        return self.params.svr_sensitivity

    @svr_sensitivity.setter
    def svr_sensitivity(self, svr_sensitivity: float):
        self.params.svr_sensitivity = svr_sensitivity
        self.non_default_attrs.add('svr_sensitivity')

    def fit(self, X, y, sample_weight=None, convert_dtype=True):
        """
        Fit the model with X and y.

        """
        self.reset_model()

        # Fit binary classifier
        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='F')

        cdef uintptr_t X_ptr = X_m.ptr
        convert_to_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype,
                                check_rows=n_rows, check_cols=1)

        cdef uintptr_t y_ptr = y_m.ptr

        # sample_weight = self._apply_class_weight(sample_weight, y_m)
        cdef uintptr_t sample_weight_ptr = <uintptr_t> nullptr
        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, check_dtype=self.dtype,
                                    convert_to_dtype=convert_to_dtype,
                                    check_rows=n_rows, check_cols=1)
            sample_weight_ptr = sample_weight_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if self.dtype == np.float32:
            self.model.float32 = new LinearSVMModel[float](
                deref(handle_),
                self.params,
                <const float*>X_ptr,
                n_rows, n_cols,
                <const float*>y_ptr,
                <const float*>sample_weight_ptr)
        elif self.dtype == np.float64:
            self.model.float64 = new LinearSVMModel[double](
                deref(handle_),
                self.params,
                <const double*>X_ptr,
                n_rows, n_cols,
                <const double*>y_ptr,
                <const double*>sample_weight_ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        del X_m
        del y_m

    def predict(self, X, convert_dtype=True) -> CumlArray:
        convert_to_dtype = self.dtype if convert_dtype else None
        X_m, n_rows, n_cols, _ = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype)

        cdef uintptr_t X_ptr = X_m.ptr

        y = CumlArray.empty(shape=(X_m.shape[0],), dtype=X_m.dtype, order='F')
        cdef uintptr_t y_ptr = y.ptr

        if self.dtype == np.float32:
            self.model.float32.predict(
                <const float*>X_ptr, n_rows, n_cols,
                <float*>y_ptr)
        elif self.dtype == np.float64:
            self.model.float64.predict(
                <const double*>X_ptr, n_rows, n_cols,
                <double*>y_ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y


class LinearSVC(LinearSVM, ClassifierMixin):

    REGISTERED_LOSSES = set([
        'hinge',
        'squared_hinge'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set classification-specific defaults
        if 'loss' not in kwargs:
            self.loss = 'squared_hinge'

    @property
    def loss(self):
        return LinearSVM.loss.__get__(self)

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Classification loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        LinearSVM.loss.__set__(self, loss)


class LinearSVR(LinearSVM, RegressorMixin):

    REGISTERED_LOSSES = set([
        'epsilon_insensitive',
        'squared_epsilon_insensitive'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set regression-specific defaults
        if 'loss' not in kwargs:
            self.loss = 'epsilon_insensitive'

    @property
    def loss(self):
        return LinearSVM.loss.__get__(self)

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Regression loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        LinearSVM.loss.__set__(self, loss)
