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

from libcpp cimport bool

ctypedef int underlying_type_t_solver

cdef extern from "cuml/decomposition/params.hpp" namespace "ML" nogil:

    ctypedef enum solver "ML::solver":
        COV_EIG_DQ "ML::solver::COV_EIG_DQ"
        COV_EIG_JACOBI "ML::solver::COV_EIG_JACOBI"

    cdef cppclass params:
        int n_rows
        int n_cols
        int gpu_id

    cdef cppclass paramsSolver(params):
        int n_rows
        int n_cols
        float tol
        int n_iterations
        int random_state
        int verbose

    cdef cppclass paramsTSVD(paramsSolver):
        int n_components
        int max_sweeps
        solver algorithm  # = solver::COV_EIG_DQ
        bool trans_input

    cdef cppclass paramsPCA(paramsTSVD):
        bool copy
        bool whiten
