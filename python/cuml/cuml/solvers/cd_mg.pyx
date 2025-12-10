#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.internals import run_in_internal_api
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
        bool normalize,
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
        bool normalize,
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
    @run_in_internal_api
    def _fit(self, uintptr_t X, uintptr_t y, uintptr_t coef_ptr, uintptr_t input_desc):
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef bool use_f32 = self.dtype == np.float32
        cdef bool fit_intercept = self.fit_intercept
        cdef bool normalize = self.normalize
        cdef int max_iter = self.max_iter
        cdef double alpha = self.alpha
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
                    deref(<vector[floatData_t*]*>X),
                    deref(<PartDescriptor*>input_desc),
                    deref(<vector[floatData_t*]*>y),
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    normalize,
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
                    deref(<vector[doubleData_t*]*>X),
                    deref(<PartDescriptor*>input_desc),
                    deref(<vector[doubleData_t*]*>y),
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    normalize,
                    max_iter,
                    alpha,
                    l1_ratio,
                    shuffle,
                    tol,
                    False
                )
        self.handle.sync()

        self.intercept_ = intercept_f32 if use_f32 else intercept_f64
        self.n_iter_ = n_iter
