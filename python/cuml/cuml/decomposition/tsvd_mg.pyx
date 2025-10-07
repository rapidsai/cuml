#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import numpy as np

import cuml.internals
from cuml.decomposition import TruncatedSVD
from cuml.decomposition.base_mg import BaseDecompositionMG
from cuml.internals.array import CumlArray

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    doubleData_t,
    floatData_t,
)
from cuml.decomposition.common cimport mg_solver, paramsTSVDMG


cdef extern from "cuml/decomposition/tsvd_mg.hpp" namespace "ML::TSVD::opg" nogil:

    cdef void fit_transform(handle_t& handle,
                            vector[floatData_t *] input_data,
                            PartDescriptor &input_desc,
                            vector[floatData_t *] trans_data,
                            PartDescriptor &trans_desc,
                            float *components,
                            float *explained_var,
                            float *explained_var_ratio,
                            float *singular_vals,
                            paramsTSVDMG &prms,
                            bool verbose) except +

    cdef void fit_transform(handle_t& handle,
                            vector[doubleData_t *] input_data,
                            PartDescriptor &input_desc,
                            vector[doubleData_t *] trans_data,
                            PartDescriptor &trans_desc,
                            double *components,
                            double *explained_var,
                            double *explained_var_ratio,
                            double *singular_vals,
                            paramsTSVDMG &prms,
                            bool verbose) except +


class TSVDMG(BaseDecompositionMG, TruncatedSVD):
    @cuml.internals.api_base_return_any_skipall
    def _mg_fit_transform(
        self, X_ptr, n_rows, n_cols, dtype, trans_ptr, input_desc_ptr, trans_desc_ptr
    ):
        # Validate and initialize parameters
        cdef paramsTSVDMG params
        params.n_components = self.n_components_
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.n_iterations = self.n_iter
        params.tol = self.tol
        if self.algorithm in ("auto", "full"):
            params.algorithm = mg_solver.COV_EIG_DQ
        elif self.algorithm == "jacobi":
            params.algorithm = mg_solver.COV_EIG_JACOBI
        else:
            raise ValueError(
                f"Expected `algorithm` to be one of ['auto', 'full', 'jacobi'], "
                f"got {self.algorithm!r}"
            )

        # Allocate output arrays
        components = CumlArray.zeros((self.n_components, n_cols), dtype=dtype)
        explained_variance = CumlArray.zeros(self.n_components, dtype=dtype)
        explained_variance_ratio = CumlArray.zeros(self.n_components, dtype=dtype)
        singular_values = CumlArray.zeros(self.n_components, dtype=dtype)

        cdef uintptr_t c_X_ptr = X_ptr
        cdef uintptr_t c_trans_ptr = trans_ptr
        cdef PartDescriptor *input_desc = <PartDescriptor*><uintptr_t>input_desc_ptr
        cdef PartDescriptor *trans_desc = <PartDescriptor*><uintptr_t>trans_desc_ptr

        cdef uintptr_t components_ptr = components.ptr
        cdef uintptr_t explained_variance_ptr = explained_variance.ptr
        cdef uintptr_t explained_variance_ratio_ptr = explained_variance_ratio.ptr
        cdef uintptr_t singular_values_ptr = singular_values.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Perform Fit
        with nogil:
            if use_float32:
                fit_transform(
                    handle_[0],
                    deref(<vector[floatData_t*]*>c_X_ptr),
                    deref(input_desc),
                    deref(<vector[floatData_t*]*>c_trans_ptr),
                    deref(trans_desc),
                    <float*> components_ptr,
                    <float*> explained_variance_ptr,
                    <float*> explained_variance_ratio_ptr,
                    <float*> singular_values_ptr,
                    params,
                    False
                )
            else:
                fit_transform(
                    handle_[0],
                    deref(<vector[doubleData_t*]*>c_X_ptr),
                    deref(input_desc),
                    deref(<vector[doubleData_t*]*>c_trans_ptr),
                    deref(trans_desc),
                    <double*> components_ptr,
                    <double*> explained_variance_ptr,
                    <double*> explained_variance_ratio_ptr,
                    <double*> singular_values_ptr,
                    params,
                    False
                )
        self.handle.sync()

        # Store results
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = singular_values
