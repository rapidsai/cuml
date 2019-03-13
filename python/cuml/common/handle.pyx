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


cimport cuml.common.cuda


cdef extern from "cuML.hpp" namespace "ML" nogil:
    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s)


# TODO: name this properly!
# TODO: add support for setting custom memory allocators
cdef class Handle:
    """
    Handle is a lightweight python wrapper around the corresponding C++ class
    of cumlHandle exposed by cuML's C++ interface. Refer to the header file
    cuML.hpp for interface level details of this struct

    Examples
    --------

    .. code-block:: python

        import cuml
        stream = cuml.cuda.Stream()
        handle = cuml.Handle()
        handle.setStream(stream)
    """
    cdef cumlHandle *h

    def __cinit__(self):
        self.h = new cumlHandle()

    def __dealloc_(self):
        del self.h

    def setStream(self, stream):
        self.h.setStream(stream.getStream())
