#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp

from cuml.linear_model.base_mg import MGFitMixin
from cuml.linear_model.linear_regression import Algo, LinearRegression

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


cdef extern from "cuml/linear_model/ols_mg.hpp" namespace "ML::OLS::opg" nogil:

    cdef void fit(handle_t& handle,
                  vector[floatData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[floatData_t *] labels,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +

    cdef void fit(handle_t& handle,
                  vector[doubleData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[doubleData_t *] labels,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  int algo,
                  bool verbose) except +


class LinearRegressionMG(MGFitMixin, LinearRegression):
    def _fit(
        self,
        uintptr_t X_ptr,
        uintptr_t y_ptr,
        n_cols,
        dtype,
        uintptr_t input_desc_ptr,
    ):
        cdef int algo = (
            Algo.EIG if self.algorithm == "auto" else Algo.parse(self.algorithm)
        )
        cdef float float_intercept
        cdef double double_intercept
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        coef = cp.zeros(n_cols, dtype=dtype)
        cdef uintptr_t coef_ptr = coef.data.ptr

        if dtype == cp.float32:

            fit(handle_[0],
                deref(<vector[floatData_t*]*>X_ptr),
                deref(<PartDescriptor*>input_desc_ptr),
                deref(<vector[floatData_t*]*>y_ptr),
                <float*>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                algo,
                False)

            self.intercept_ = float_intercept
        else:

            fit(handle_[0],
                deref(<vector[doubleData_t*]*>X_ptr),
                deref(<PartDescriptor*>input_desc_ptr),
                deref(<vector[doubleData_t*]*>y_ptr),
                <double*>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

        self.coef_ = coef
