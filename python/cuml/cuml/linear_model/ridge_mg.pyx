#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from cuml.internals import run_in_internal_context
from cuml.linear_model import Ridge
from cuml.linear_model.base_mg import MGFitMixin

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport *


cdef extern from "cuml/linear_model/ridge_mg.hpp" namespace "ML::Ridge::opg" nogil:

    cdef void fit(handle_t& handle,
                  vector[floatData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[floatData_t *] labels,
                  float *alpha,
                  int n_alpha,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +

    cdef void fit(handle_t& handle,
                  vector[doubleData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[doubleData_t *] labels,
                  double *alpha,
                  int n_alpha,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +


class RidgeMG(MGFitMixin, Ridge):
    @run_in_internal_context
    def _fit(self, X, y, coef_ptr, input_desc):
        # Validate alpha
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        # Validate and select solver
        SUPPORTED_SOLVERS = ["auto", "eig", "svd"]
        if (solver := self.solver) not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"Expected `solver` to be one of {SUPPORTED_SOLVERS}, got {solver!r}"
            )

        if solver == "eig":
            if self.n_features_in_ == 1:
                raise ValueError(
                    "solver='eig' doesn't support X with 1 column, please select "
                    "solver='svd' instead"
                )
        elif solver == "auto":
            solver = "svd" if self.n_features_in_ == 1 else "eig"

        cdef int algo = {"svd": 0, "eig": 1}[solver]

        cdef float intercept_f32
        cdef double intercept_f64
        cdef float alpha_f32 = self.alpha
        cdef double alpha_f64 = self.alpha
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef bool use_float32 = self.dtype == np.float32

        if use_float32:
            fit(handle_[0],
                deref(<vector[floatData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>input_desc),
                deref(<vector[floatData_t*]*><uintptr_t>y),
                &alpha_f32,
                1,
                <float*><size_t>coef_ptr,
                <float*>&intercept_f32,
                <bool>self.fit_intercept,
                algo,
                False)
        else:
            fit(handle_[0],
                deref(<vector[doubleData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>input_desc),
                deref(<vector[doubleData_t*]*><uintptr_t>y),
                &alpha_f64,
                1,
                <double*><size_t>coef_ptr,
                <double*>&intercept_f64,
                <bool>self.fit_intercept,
                algo,
                False)

        self.handle.sync()

        self.solver_ = solver
        self.intercept_ = (intercept_f32 if use_float32 else intercept_f64)
