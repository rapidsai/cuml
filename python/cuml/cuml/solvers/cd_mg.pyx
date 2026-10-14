#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.linear_model.base_mg import MGFitMixin
from cuml.solvers import CD

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


cdef extern from "cuml/solvers/cd_mg.hpp" namespace "ML::CD::opg" nogil:
    cdef int fit(
        handle_t& handle,
        vector[floatData_t *] input_data,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        float *intercept,
        bool fit_intercept,
        int epochs,
        float alpha,
        float l1_ratio,
        bool shuffle,
        float tol,
        bool verbose
    ) except +

    cdef int fit(
        handle_t& handle,
        vector[doubleData_t *] input_data,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        double *intercept,
        bool fit_intercept,
        int epochs,
        double alpha,
        double l1_ratio,
        bool shuffle,
        double tol,
        bool verbose
    ) except +


class CDMG(MGFitMixin, CD):
    """
    Cython class for MNMG code usage. Not meant for end user consumption.
    """
    def _fit(
        self,
        uintptr_t X_ptr,
        uintptr_t y_ptr,
        n_cols,
        dtype,
        uintptr_t input_desc_ptr,
    ):
        coef = cp.zeros(n_cols, dtype=dtype)
        cdef uintptr_t coef_ptr = coef.data.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef bool use_f32 = dtype == cp.float32
        cdef bool fit_intercept = self.fit_intercept
        cdef int max_iter = self.max_iter
        cdef double alpha = (
            self.alpha if cp.isscalar(self.alpha) else self.alpha.item()
        )
        cdef double l1_ratio = self.l1_ratio
        cdef bool shuffle = self.shuffle
        cdef double tol = self.tol
        cdef float intercept_f32
        cdef double intercept_f64
        cdef int n_iter

        with nogil:
            if use_f32:
                n_iter = fit(
                    handle_[0],
                    deref(<vector[floatData_t*]*>X_ptr),
                    deref(<PartDescriptor*>input_desc_ptr),
                    deref(<vector[floatData_t*]*>y_ptr),
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    max_iter,
                    <float>alpha,
                    <float>l1_ratio,
                    shuffle,
                    <float>tol,
                    False
                )
            else:
                n_iter = fit(
                    handle_[0],
                    deref(<vector[doubleData_t*]*>X_ptr),
                    deref(<PartDescriptor*>input_desc_ptr),
                    deref(<vector[doubleData_t*]*>y_ptr),
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    max_iter,
                    alpha,
                    l1_ratio,
                    shuffle,
                    tol,
                    False
                )
        self.handle.sync()

        self.coef_ = coef
        self.intercept_ = intercept_f32 if use_f32 else intercept_f64
        self.n_iter_ = n_iter
