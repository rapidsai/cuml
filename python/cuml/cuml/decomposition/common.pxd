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
