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

import functools
from libcpp.string cimport string

cdef extern from "common/nvtx.hpp" namespace "ML":

    void PUSH_RANGE(string msg)

    void POP_RANGE()


def nvtx_range_push(msg: str):
    """Create a NVTX range with name `msg`"""
    cdef string s = msg.encode("UTF-8")
    PUSH_RANGE(s.c_str())


def nvtx_range_pop():
    """End a NVTX range"""
    POP_RANGE()


def nvtx_range_wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nvtx_range_push(func.__name__)
        result = func(*args, **kwargs)
        nvtx_range_pop()
        return result
    return wrapper


class CudaRuntimeError(RuntimeError):
    def __init__(self, extraMsg=None):
        cdef _Error e = cudaGetLastError()
        cdef bytes errMsg = cudaGetErrorString(e)
        cdef bytes errName = cudaGetErrorName(e)
        msg = "Error! %s reason='%s'" % (errName.decode(), errMsg.decode())
        if extraMsg is not None:
            msg += " extraMsg='%s'" % extraMsg
        super(CudaRuntimeError, self).__init__(msg)


cdef class Stream:
    """
    Stream represents a thin-wrapper around cudaStream_t and its operations.

    Examples
    --------

    .. code-block:: python

        import cuml
        stream = cuml.cuda.Stream()
        stream.sync()
        del stream  # optional!
    """

    # NOTE:
    # If we store _Stream directly, this always leads to the following error:
    #   "Cannot convert Python object to '_Stream'"
    # I was unable to find a good solution to this in reasonable time. Also,
    # since cudaStream_t is a pointer anyways, storing it as an integer should
    # be just fine (although, that certainly is ugly and hacky!).
    cdef size_t s

    def __cinit__(self):
        if self.s != 0:
            return
        cdef _Stream stream
        cdef _Error e = cudaStreamCreate(&stream)
        if e != 0:
            raise CudaRuntimeError("Stream create")
        self.s = <size_t>stream

    def __dealloc__(self):
        self.sync()
        cdef _Stream stream = <_Stream>self.s
        cdef _Error e = cudaStreamDestroy(stream)
        if e != 0:
            raise CudaRuntimeError("Stream destroy")

    def sync(self):
        """
        Synchronize on the cudastream owned by this object. Note that this
        could raise exception due to issues with previous asynchronous
        launches
        """
        cdef _Stream stream = <_Stream>self.s
        cdef _Error e = cudaStreamSynchronize(stream)
        if e != 0:
            raise CudaRuntimeError("Stream sync")

    def getStream(self):
        return self.s
