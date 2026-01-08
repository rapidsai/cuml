#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.decomposition import TruncatedSVD
from cuml.decomposition.base_mg import BaseDecompositionMG
from cuml.internals import get_handle, run_in_internal_context
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
                            bool verbose,
                            bool flip_signs_based_on_U) except +

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
                            bool verbose,
                            bool flip_signs_based_on_U) except +


class TSVDMG(BaseDecompositionMG, TruncatedSVD):
    @run_in_internal_context
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

        if self.n_components > n_cols:
            raise ValueError(
                f"`n_components` ({self.n_components}) must be <= than the "
                f"number of features in X ({n_cols})"
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
        handle = get_handle(model=self)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef bool flip_signs_based_on_U = self._u_based_sign_flip

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
                    False,
                    flip_signs_based_on_U
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
                    False,
                    flip_signs_based_on_U
                )
        handle.sync()

        # Store results
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = singular_values
