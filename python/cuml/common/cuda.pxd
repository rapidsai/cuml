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


# Populate this with more typedef's (eg: events) as and when needed
cdef extern from * nogil:
    ctypedef void* _Stream "cudaStream_t"
    ctypedef int   _Error  "cudaError_t"


# Populate this with more runtime api method declarations as and when needed
cdef extern from "cuda_runtime_api.h" nogil:
    _Error cudaStreamCreate(_Stream* s)
    _Error cudaStreamDestroy(_Stream s)
    _Error cudaStreamSynchronize(_Stream s)
    _Error cudaGetLastError()
    const char* cudaGetErrorString(_Error e)
    const char* cudaGetErrorName(_Error e)
