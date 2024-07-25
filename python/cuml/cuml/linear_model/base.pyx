#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import cuml.internals
from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')
from cuml.internals.safe_imports import gpu_only_import
cp = gpu_only_import('cupy')

from cuml.internals.safe_imports import gpu_only_import_from
cuda = gpu_only_import_from('numba', 'cuda')

from libc.stdint cimport uintptr_t

from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array
from cuml.common.doc_utils import generate_docstring
from cuml.internals.api_decorators import enable_device_interop


IF GPUBUILD == 1:
    from pylibraft.common.handle cimport handle_t
    cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

        cdef void gemmPredict(handle_t& handle,
                              const float *input,
                              size_t _n_rows,
                              size_t _n_cols,
                              const float *coef,
                              float intercept,
                              float *preds) except +

        cdef void gemmPredict(handle_t& handle,
                              const double *input,
                              size_t _n_rows,
                              size_t _n_cols,
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
        X_m, _n_rows, _n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_features_in_)
        cdef uintptr_t _X_ptr = X_m.ptr
        cdef uintptr_t _coef_ptr = self.coef_.ptr

        preds = CumlArray.zeros(_n_rows, dtype=dtype, index=X_m.index)
        cdef uintptr_t _preds_ptr = preds.ptr

        IF GPUBUILD == 1:
            cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
            if dtype.type == np.float32:
                gemmPredict(handle_[0],
                            <float*>_X_ptr,
                            <size_t>_n_rows,
                            <size_t>_n_cols,
                            <float*>_coef_ptr,
                            <float>self.intercept_,
                            <float*>_preds_ptr)
            else:
                gemmPredict(handle_[0],
                            <double*>_X_ptr,
                            <size_t>_n_rows,
                            <size_t>_n_cols,
                            <double*>_coef_ptr,
                            <double>self.intercept_,
                            <double*>_preds_ptr)

        self.handle.sync()

        return preds
