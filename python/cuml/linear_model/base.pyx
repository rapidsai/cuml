#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import cuml.internals
import numpy as np
import cupy as cp
import warnings

from numba import cuda
from collections import defaultdict

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.internals.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from pylibraft.common.handle cimport handle_t
from cuml.internals.api_decorators import enable_device_interop

cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

    cdef void gemmPredict(handle_t& handle,
                          const float *input,
                          size_t n_rows,
                          size_t n_cols,
                          const float *coef,
                          float intercept,
                          float *preds) except +

    cdef void gemmPredict(handle_t& handle,
                          const double *input,
                          size_t n_rows,
                          size_t n_cols,
                          const double *coef,
                          double intercept,
                          double *preds) except +


class LinearPredictMixin:
    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    @cuml.internals.api_base_return_array_skipall
    @enable_device_interop
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.

        """
        self.dtype = self.coef_.dtype

        if self.coef_ is None:
            raise ValueError(
                "LinearModel.predict() cannot be called before fit(). "
                "Please fit the model first."
            )
        self.dtype = self.coef_.dtype
        if len(self.coef_.shape) == 2 and self.coef_.shape[0] > 1:
            # Handle multi-target prediction in Python.
            coef_arr = CumlArray.from_input(self.coef_).to_output('array')
            X_arr = CumlArray.from_input(
                X,
                check_dtype=self.dtype,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                check_cols=self.n_features_in_
            ).to_output('array')
            intercept_arr = CumlArray.from_input(
                self.intercept_
            ).to_output('array')
            preds_arr = X_arr @ coef_arr + intercept_arr
            return preds_arr

        # Handle single-target prediction in C++
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_features_in_)
        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t coef_ptr = self.coef_.ptr

        preds = CumlArray.zeros(n_rows, dtype=dtype, index=X_m.index)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            gemmPredict(handle_[0],
                        <float*>X_ptr,
                        <size_t>n_rows,
                        <size_t>n_cols,
                        <float*>coef_ptr,
                        <float>self.intercept_,
                        <float*>preds_ptr)
        else:
            gemmPredict(handle_[0],
                        <double*>X_ptr,
                        <size_t>n_rows,
                        <size_t>n_cols,
                        <double*>coef_ptr,
                        <double>self.intercept_,
                        <double*>preds_ptr)

        self.handle.sync()

        return preds
