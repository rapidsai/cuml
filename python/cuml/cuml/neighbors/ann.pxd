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

# distutils: language = c++


from libc.stdint cimport uintptr_t
from libcpp cimport bool


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    cdef cppclass knnIndex:
        pass

    cdef cppclass knnIndexParam:
        pass

    cdef cppclass IVFParam(knnIndexParam):
        int nlist
        int nprobe

    cdef cppclass IVFFlatParam(IVFParam):
        pass

    cdef cppclass IVFPQParam(IVFParam):
        int M
        int n_bits
        bool usePrecomputedTables


cdef check_algo_params(algo, params)


cdef build_ivfflat_algo_params(params, automated)


cdef build_ivfpq_algo_params(params, automated, additional_info)


cdef build_algo_params(algo, params, additional_info)


cdef destroy_algo_params(ptr)
