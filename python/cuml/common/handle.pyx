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


cdef extern from "cuML.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s)
        void setDeviceAllocator(shared_ptr[deviceAllocator] a)


cdef extern from "common/rmmAllocatorAdapter.hpp" namespace "ML" nogil:
    cdef cppclass rmmAllocatorAdapter(deviceAllocator):
        pass


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
        stream = cuml.common.cuda.Stream()
        handle = cuml.common.handle.Handle()
        handle.setStream(stream)

        # call ML algos here

        # final synchronization of all work launched/dependent on this stream
        stream.sync()
        del handle  # optional!
    """

    # ML::cumlHandle doesn't have copy operator. So, use pointer for the object
    cdef cumlHandle *h

    def __cinit__(self):
        self.h = new cumlHandle()

    def __dealloc_(self):
        del self.h

    def setStream(self, stream):
        cdef size_t s = <size_t>stream.getStream()
        self.h.setStream(<cuml.common.cuda._Stream>s)

    # TODO: in future, maybe we should just enable RMM by default!?
    def enableRMM(self):
        """
        Enables to use RMM as the allocator for all device memory allocations
        inside cuML C++ world. Currently, there are only 2 kinds of allocators.
        First, the usual cudaMalloc/Free, which is the default for cumlHandle.
        Second, the allocator based on RMM. So, this function, basically makes
        the cumlHandle use a more efficient allocator, instead of the default.
        """
        cdef shared_ptr[deviceAllocator] rmmAlloc = shared_ptr[deviceAllocator](new rmmAllocatorAdapter())
        self.h.setDeviceAllocator(rmmAlloc)
