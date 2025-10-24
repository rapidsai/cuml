#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from libcpp cimport bool


cdef extern from "cuml/decomposition/params.hpp" namespace "ML" nogil:

    enum solver "ML::solver":
        COV_EIG_DQ "ML::solver::COV_EIG_DQ"
        COV_EIG_JACOBI "ML::solver::COV_EIG_JACOBI"

    cdef cppclass params:
        size_t n_rows
        size_t n_cols

    cdef cppclass paramsSolver(params):
        float tol
        unsigned n_iterations
        int verbose

    cdef cppclass paramsTSVD(paramsSolver):
        size_t n_components
        solver algorithm

    cdef cppclass paramsPCA(paramsTSVD):
        bool copy
        bool whiten

    enum mg_solver "ML::mg_solver":
        COV_EIG_DQ "ML::mg_solver::COV_EIG_DQ"
        COV_EIG_JACOBI "ML::mg_solver::COV_EIG_JACOBI"

    cdef cppclass paramsTSVDMG(paramsSolver):
        size_t n_components
        mg_solver algorithm

    cdef cppclass paramsPCAMG(paramsTSVDMG):
        bool copy
        bool whiten
