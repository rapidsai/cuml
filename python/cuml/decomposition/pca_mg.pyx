#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
import cudf
import numpy as np
from enum import IntEnum

import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.array import CumlArray
import cuml.common.opg_data_utils_mg as opg

import cuml.internals
from cuml.common.base import Base
from cuml.raft.common.handle cimport handle_t
from cuml.decomposition.utils cimport paramsSolver
from cuml.common.opg_data_utils_mg cimport *

from cuml.decomposition import PCA
from cuml.decomposition.base_mg import BaseDecompositionMG


ctypedef int underlying_type_t_solver


cdef extern from "cuml/decomposition/pca_mg.hpp" namespace "ML":

    ctypedef enum mg_solver "ML::mg_solver":
        COV_EIG_DQ "ML::mg_solver::COV_EIG_DQ"
        COV_EIG_JACOBI "ML::mg_solver::COV_EIG_JACOBI"
        QR "ML::mg_solver::QR"

    cdef cppclass paramsTSVDMG(paramsSolver):
        int n_components
        int max_sweeps
        mg_solver algorithm  # = solver::COV_EIG_DQ
        bool trans_input

    cdef cppclass paramsPCAMG(paramsTSVDMG):
        bool copy
        bool whiten


cdef extern from "cuml/decomposition/pca_mg.hpp" namespace "ML::PCA::opg":

    cdef void fit(handle_t& handle,
                  vector[floatData_t *] input_data,
                  PartDescriptor &input_desc,
                  float *components,
                  float *explained_var,
                  float *explained_var_ratio,
                  float *singular_vals,
                  float *mu,
                  float *noise_vars,
                  paramsPCAMG &prms,
                  bool verbose) except +

    cdef void fit(handle_t& handle,
                  vector[doubleData_t *] input_data,
                  PartDescriptor &input_desc,
                  double *components,
                  double *explained_var,
                  double *explained_var_ratio,
                  double *singular_vals,
                  double *mu,
                  double *noise_vars,
                  paramsPCAMG &prms,
                  bool verbose) except +


class MGSolver(IntEnum):
    COV_EIG_DQ = <underlying_type_t_solver> mg_solver.COV_EIG_DQ
    COV_EIG_JACOBI = <underlying_type_t_solver> mg_solver.COV_EIG_JACOBI
    QR = <underlying_type_t_solver> mg_solver.QR


class PCAMG(BaseDecompositionMG, PCA):

    def __init__(self, **kwargs):
        super(PCAMG, self).__init__(**kwargs)

    def _get_algorithm_c_name(self, algorithm):
        algo_map = {
            'full': MGSolver.COV_EIG_DQ,
            'auto': MGSolver.COV_EIG_JACOBI,
            # 'arpack': NOT_SUPPORTED,
            # 'randomized': NOT_SUPPORTED,
        }
        if algorithm not in algo_map:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        return algo_map[algorithm]

    def _build_params(self, n_rows, n_cols):
        cpdef paramsPCAMG *params = new paramsPCAMG()
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        params.algorithm = <mg_solver> (<underlying_type_t_solver> (
            self.c_algorithm))

        return <size_t>params

    @cuml.internals.api_base_return_any_skipall
    def _call_fit(self, X, rank, part_desc, arg_params):

        cdef uintptr_t comp_ptr = self.components_.ptr
        cdef uintptr_t explained_var_ptr = self.explained_variance_.ptr
        cdef uintptr_t explained_var_ratio_ptr = \
            self.explained_variance_ratio_.ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_.ptr
        cdef uintptr_t mean_ptr = self.mean_.ptr
        cdef uintptr_t noise_vars_ptr = self.noise_variance_.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef paramsPCAMG *params = <paramsPCAMG*><size_t>arg_params

        if self.dtype == np.float32:

            fit(handle_[0],
                deref(<vector[floatData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>part_desc),
                <float*> comp_ptr,
                <float*> explained_var_ptr,
                <float*> explained_var_ratio_ptr,
                <float*> singular_vals_ptr,
                <float*> mean_ptr,
                <float*> noise_vars_ptr,
                deref(params),
                False)
        else:

            fit(handle_[0],
                deref(<vector[doubleData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>part_desc),
                <double*> comp_ptr,
                <double*> explained_var_ptr,
                <double*> explained_var_ratio_ptr,
                <double*> singular_vals_ptr,
                <double*> mean_ptr,
                <double*> noise_vars_ptr,
                deref(params),
                False)

        self.handle.sync()
