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


import cuml
from libcpp.memory cimport shared_ptr
from cuml.common.cuda cimport _Stream, _Error, cudaStreamSynchronize


#cdef extern from "common/rmmAllocatorAdapter.hpp" namespace "ML" nogil:
#    cdef cppclass rmmAllocatorAdapter(deviceAllocator):
#        pass


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
        handle.enableRMM()   # Enable RMM as the device-side allocator

        # call ML algos here

        # final sync of all work launched in the stream of this handle
        # this is same as `cuml.cuda.Stream.sync()` call, but safer in case
        # the default stream inside the `cumlHandle` is being used
        handle.sync()
        del handle  # optional!
    """

    # ML::cumlHandle doesn't have copy operator. So, use pointer for the object
    # python world cannot access to this raw object directly, hence use 'size_t'!
    cdef size_t h

    def __cinit__(self):
        self.h = <size_t>(new cumlHandle())

    def __dealloc_(self):
        h_ = <cumlHandle*>self.h
        del h_

    def setStream(self, stream):
        cdef size_t s = <size_t>stream.getStream()
        cdef cumlHandle* h_ = <cumlHandle*>self.h
        h_.setStream(<_Stream>s)

    # TODO: in future, we should just enable RMM by default
    def enableRMM(self):
        """
        Enables to use RMM as the allocator for all device memory allocations
        inside cuML C++ world. Currently, there are only 2 kinds of allocators.
        First, the usual cudaMalloc/Free, which is the default for cumlHandle.
        Second, the allocator based on RMM. So, this function, basically makes
        the cumlHandle use a more efficient allocator, instead of the default.
        """
#        cdef shared_ptr[deviceAllocator] rmmAlloc = shared_ptr[deviceAllocator](new rmmAllocatorAdapter())
#        cdef cumlHandle* h_ = <cumlHandle*>self.h
#        h_.setDeviceAllocator(rmmAlloc)

    def sync(self):
        """
        Issues a sync on the stream set for this handle.

        Once we make `cuml.cuda.Stream` as a mandatory option for creating `cuml.Handle`,
        this should go away!
        """
        cdef cumlHandle* h_ = <cumlHandle*>self.h
        cdef _Stream stream = h_.getStream()
        cdef _Error e = cudaStreamSynchronize(stream)
        if e != 0:
            raise cuml.cuda.CudaRuntimeError("Stream sync")

    def getHandle(self):
        return self.h
