#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from libcpp cimport bool
from cuml.decomposition.utils cimport *

cdef extern from "cuml/decomposition/params.hpp" namespace "ML" nogil:

    ctypedef enum mg_solver "ML::mg_solver":
        COV_EIG_DQ "ML::mg_solver::COV_EIG_DQ"
        COV_EIG_JACOBI "ML::mg_solver::COV_EIG_JACOBI"
        QR "ML::mg_solver::QR"

    cdef cppclass paramsTSVDMG(paramsSolver):
        size_t n_components
        mg_solver algorithm  # = solver::COV_EIG_DQ

    cdef cppclass paramsPCAMG(paramsTSVDMG):
        bool copy
        bool whiten
