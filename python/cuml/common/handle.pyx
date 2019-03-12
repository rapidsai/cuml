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


cdef extern from "cuda_runtime.h":
    cdef struct cudaStream_t:
        pass


cdef extern from *: #"cuML_api.h":
    cdef struct cumlHandle_t:
         void* ptr

cdef extern from "cuML_api.h":
    cdef enum cumlError_t:
        pass

    cumlError_t cumlCreate(cumlHandle_t* handle)
    cumlError_t cumlSetStream(cumlHandle_t handle, cudaStream_t stream)
    cumlError_t cumlDestroy(cumlHandle_t handle)


cdef class cumlHandle:
    """
    cumlHandle is a lightweight wrapper around the corresponding C-struct
    (and APIs) exposed by cuML's C++ interface. Refer to the header files
    cuML_api.h and cuML.hpp for interface level details of this struct
    """
    cdef cumlHandle_t *handle

    def __cinit__(self):
        cdef cumlError_t err = cumlCreate(self.handle)
        # TODO: clean this!
        if err != 0:
            raise Exception(err)

    # def __dealloc__(self):
    #     cdef cumlError_t err = cumlDestroy(*self.handle)
    #     # TODO: clean this!
    #     if err != 0:
    #         print("Warning: failed to destroy cumlHandle!")

    # def setStream(self, stream):
    #     cdef cumlError_t err = cumlSetStream(*self.handle, stream)
    #     # TODO: clean this!
    #     if err != 0:
    #         raise Exception(err)
