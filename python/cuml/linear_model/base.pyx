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

from cuml.common.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from pylibraft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.input_utils import input_to_cupy_array

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
    def _predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.

        """
        self.dtype = self.coef_.dtype

        if self.coef_ is None:
            raise ValueError(
                "LinearModel.predict() cannot be called before fit(). "
                "Please fit the model first."
            )
        n_targets = self.coef_.shape[0]
        if len(self.coef_.shape) == 2 and n_targets > 1:
            # Handle multi-target prediction in Python.
            coef_cp = input_to_cupy_array(self.coef_).array
            X_cp = input_to_cupy_array(
                X,
                check_dtype=self.dtype,
                convert_to_dtype=(self.dtype if convert_dtype else None),
                check_cols=self.n_features_in_
            ).array
            intercept_cp = input_to_cupy_array(self.intercept_).array
            preds_cp = X_cp @ coef_cp + intercept_cp
            # preds = input_to_cuml_array(preds_cp).array # TODO:remove
            return preds_cp

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
