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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.memory cimport shared_ptr
cimport cuml.common.cuda


cdef extern from "cuml/cuml.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        cumlHandle(int ns) except +
        void setStream(cuml.common.cuda._Stream s) except +
        void setDeviceAllocator(shared_ptr[deviceAllocator] a) except +
        cuml.common.cuda._Stream getStream() except +
        int getNumInternalStreams() except +
