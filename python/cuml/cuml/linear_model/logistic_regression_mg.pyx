#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.linear_model import LogisticRegression
from cuml.linear_model.base_mg import MGFitMixin

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport (
    PartDescriptor,
    doubleData_t,
    floatData_t,
)
from cuml.solvers.qn cimport init_qn_params, qn_params


cdef extern from "cuml/linear_model/qn_mg.hpp" namespace "ML::GLM::opg" nogil:
    cdef void qnFit(
        handle_t& handle,
        vector[floatData_t *] input_data,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool X_col_major,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFit(
        handle_t& handle,
        vector[doubleData_t *] input_data,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool X_col_major,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +

    cdef vector[float] getUniquelabelsMG(
        const handle_t& handle,
        PartDescriptor &input_desc,
        vector[floatData_t*] labels) except+

    cdef vector[double] getUniquelabelsMG(
        const handle_t& handle,
        PartDescriptor &input_desc,
        vector[doubleData_t*] labels) except+

    cdef void qnFitSparse(
        handle_t& handle,
        vector[floatData_t *] input_values,
        int *input_cols,
        int *input_row_ids,
        int X_nnz,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[doubleData_t *] input_values,
        int *input_cols,
        int *input_row_ids,
        int X_nnz,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[floatData_t *] input_values,
        int64_t *input_cols,
        int64_t *input_row_ids,
        int64_t X_nnz,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef void qnFitSparse(
        handle_t& handle,
        vector[doubleData_t *] input_values,
        int64_t *input_cols,
        int64_t *input_row_ids,
        int64_t X_nnz,
        PartDescriptor &input_desc,
        vector[doubleData_t *] labels,
        double *coef,
        const qn_params& pams,
        bool standardization,
        int n_classes,
        double *f,
        int *num_iters) except +


class LogisticRegressionMG(MGFitMixin, LogisticRegression):

    def __init__(self, *, standardization=False, _convert_index=False, **kwargs):
        super(LogisticRegressionMG, self).__init__(**kwargs)
        self.standardization = standardization
        self._convert_index = _convert_index

    def fit(self, input_data, n_rows, n_cols, parts_rank_size, rank, convert_dtype=False):
        if len(input_data) != 1:
            raise ValueError(
                f"Currently support only one (X, y) pair in the list. "
                f"Received {len(input_data)} pairs."
            )
        super().fit(
            input_data,
            n_rows,
            n_cols,
            parts_rank_size,
            rank,
            order="C",
            convert_index=self._convert_index,
        )

    @cuml.internals.api_base_return_any_skipall
    def _fit(self, X, uintptr_t y, uintptr_t coef_ptr, uintptr_t input_desc):
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Determine the number of classes in y
        if self.dtype == np.float32:
            classes = getUniquelabelsMG(
                handle_[0],
                deref(<PartDescriptor*>input_desc),
                deref(<vector[floatData_t*]*>y)
            )
            self.classes_ = np.sort(classes).astype(np.float32)
        elif self.dtype == np.float64:
            classes = getUniquelabelsMG(
                handle_[0],
                deref(<PartDescriptor*>input_desc),
                deref(<vector[doubleData_t*]*>y)
            )
            self.classes_ = np.sort(classes)
        else:
            raise ValueError(
                "dtypes other than float32 and float64 are currently not supported yet."
            )

        cdef int n_classes = len(self.classes_)

        # Validate and initialize parameters
        l1_strength, l2_strength = self._get_l1_l2_strength()
        cdef qn_params params
        init_qn_params(
            params,
            n_classes=n_classes,
            loss=("softmax" if n_classes > 2 else "sigmoid"),
            fit_intercept=self.fit_intercept,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            max_iter=self.max_iter,
            tol=self.tol,
            delta=None,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=5,
            penalty_normalized=True,
            verbose=self.verbose,
        )

        # Allocate outputs
        coef_n_cols = n_classes if n_classes > 2 else 1
        coef_n_rows = self.n_cols + 1 if self.fit_intercept else self.n_cols
        coef = CumlArray.zeros((coef_n_rows, coef_n_cols), dtype=self.dtype, order="C")

        # Prepare for fit
        cdef uintptr_t X_ptr = 0, X_cols_ptr = 0, X_rows_ptr = 0
        cdef int64_t X_nnz = 0
        cdef bool X_is_f32 = self.dtype == np.float32
        cdef bool X_index_is_i32 = False
        cdef bool X_is_dense = isinstance(X, int)
        if X_is_dense:
            X_ptr = X
        else:
            X_ptr, X_cols_ptr, X_rows_ptr, X_nnz = X
            X_index_is_i32 = self.index_dtype == np.int32

        coef_ptr = coef.ptr
        cdef bool standardization = self.standardization
        cdef float objective_f32
        cdef double objective_f64
        cdef int n_iter

        # Perform fit
        with nogil:
            if X_is_dense:
                if X_is_f32:
                    qnFit(
                        handle_[0],
                        deref(<vector[floatData_t*]*>X_ptr),
                        deref(<PartDescriptor*>input_desc),
                        deref(<vector[floatData_t*]*>y),
                        <float*>coef_ptr,
                        params,
                        False,
                        standardization,
                        n_classes,
                        &objective_f32,
                        &n_iter
                    )
                else:
                    qnFit(
                        handle_[0],
                        deref(<vector[doubleData_t*]*>X_ptr),
                        deref(<PartDescriptor*>input_desc),
                        deref(<vector[doubleData_t*]*>y),
                        <double*>coef_ptr,
                        params,
                        False,
                        standardization,
                        n_classes,
                        &objective_f64,
                        &n_iter
                    )
            else:
                if X_is_f32:
                    if X_index_is_i32:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[floatData_t*]*>X_ptr),
                            <int*>X_cols_ptr,
                            <int*>X_rows_ptr,
                            <int>X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[floatData_t*]*>y),
                            <float*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f32,
                            &n_iter,
                        )
                    else:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[floatData_t*]*>X_ptr),
                            <int64_t *>X_cols_ptr,
                            <int64_t *>X_rows_ptr,
                            X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[floatData_t*]*>y),
                            <float*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f32,
                            &n_iter,
                        )
                else:
                    if X_index_is_i32:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[doubleData_t*]*>X_ptr),
                            <int*>X_cols_ptr,
                            <int*>X_rows_ptr,
                            <int>X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[doubleData_t*]*>y),
                            <double*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f64,
                            &n_iter,
                        )
                    else:
                        qnFitSparse(
                            handle_[0],
                            deref(<vector[doubleData_t*]*>X_ptr),
                            <int64_t *>X_cols_ptr,
                            <int64_t *>X_rows_ptr,
                            X_nnz,
                            deref(<PartDescriptor*>input_desc),
                            deref(<vector[doubleData_t*]*>y),
                            <double*>coef_ptr,
                            params,
                            standardization,
                            n_classes,
                            &objective_f64,
                            &n_iter,
                        )

        self.handle.sync()

        # Postprocess coef into coef_ and intercept_
        coef_cp = coef.to_output("cupy")
        if self.fit_intercept:
            intercept = CumlArray(data=coef_cp[-1])
            coef = CumlArray(data=coef_cp[:-1].T)
        else:
            if n_classes <= 2:
                intercept = CumlArray.zeros(shape=1)
            else:
                intercept = CumlArray.zeros(shape=n_classes)
            coef = CumlArray(data=coef_cp.T)

        # Store fitted attributes
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = np.array([n_iter])

        return self
