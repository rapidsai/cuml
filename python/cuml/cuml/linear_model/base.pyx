#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import numpy as np

import cuml.internals

from libc.stdint cimport uintptr_t

from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array

from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

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
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.
        """
        self.dtype = self.coef_.dtype

        if self.coef_ is None:
            raise ValueError(
                "LinearModel.predict() cannot be called before fit(). "
                "Please fit the model first."
            )

        multi_target = (len(self.coef_.shape) == 2 and self.coef_.shape[0] > 1)
        if multi_target:
            return self._predict_multi_target(X, convert_dtype)
        else:
            return self._predict_single_target(X, convert_dtype)

    def _predict_multi_target(self, X, convert_dtype=True) -> CumlArray:
        """
        Predict for multi-target case.
        """
        coef_arr = CumlArray.from_input(self.coef_).to_output('array')
        X_m = CumlArray.from_input(
            X,
            check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_cols=self.n_features_in_
        )
        X_arr = X_m.to_output('array')
        if isinstance(self.intercept_, (int, float, np.number)):
            intercept_ = self.intercept_
        else:
            intercept_ = CumlArray.from_input(self.intercept_).to_output('array')

        preds_arr = X_arr @ coef_arr.T + intercept_

        # Preserve the input's index in the prediction output
        return CumlArray(preds_arr, index=X_m.index)

    def _predict_single_target(self, X, convert_dtype=True) -> CumlArray:
        """
        Predict for single-target case using C++ implementation.
        """
        X_m, _n_rows, _n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_features_in_)
        cdef uintptr_t _X_ptr = X_m.ptr
        cdef uintptr_t _coef_ptr = self.coef_.ptr

        # Assume intercept is a scalar for single-target case
        if isinstance(self.intercept_, CumlArray):
            try:
                intercept = self.intercept_.item()
            except (ValueError, TypeError):
                raise ValueError(
                    "For single-target prediction, intercept must be a scalar "
                    "or size-1 array"
                )
        else:
            intercept = self.intercept_

        # If coef_ is 2D with shape (1, n_features), ensure predictions are also 2D
        if len(self.coef_.shape) == 2:
            # coef_ is 2D with shape (1, n_features), but this function only
            # handles single-target prediction. For multi-target prediction, use
            # _predict_multi_target() instead.
            assert self.coef_.shape == (1, _n_cols)
            preds = CumlArray.zeros((_n_rows, 1), dtype=dtype, index=X_m.index)
        else:
            preds = CumlArray.zeros(_n_rows, dtype=dtype, index=X_m.index)
        cdef uintptr_t _preds_ptr = preds.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if dtype.type == np.float32:
            gemmPredict(handle_[0],
                        <float*>_X_ptr,
                        <size_t>_n_rows,
                        <size_t>_n_cols,
                        <float*>_coef_ptr,
                        <float>intercept,
                        <float*>_preds_ptr)
        else:
            gemmPredict(handle_[0],
                        <double*>_X_ptr,
                        <size_t>_n_rows,
                        <size_t>_n_cols,
                        <double*>_coef_ptr,
                        <double>intercept,
                        <double*>_preds_ptr)

        self.handle.sync()

        return preds
