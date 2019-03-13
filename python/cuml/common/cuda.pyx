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


class CudaRtError(RuntimeError):
    def __init__(self, extraMsg=None):
        cdef _Error e = cudaGetLastError()
        cdef bytes errMsg = cudaGetErrorString(e)
        cdef bytes errName = cudaGetErrorName(e)
        msg = "Error! %s reason='%s'" % (errName.decode(), errMsg.decode())
        if extraMsg is not None:
            msg += " extraMsg='%s'" % extraMsg
        super(CudaRtError, self).__init__(msg)


cdef class Stream:
    """
    Stream represents a thin-wrapper around cudaStream_t and its operations.
    """

    cdef _Stream s

    def __cinit__(self):
        cdef _Stream stream;
        cdef _Error e = cudaStreamCreate(&stream)
        if e != 0:
            raise CudaRtError("Stream create")
        self.s = stream

    def __dealloc__(self):
        cdef _Error e = cudaStreamDestroy(self.s)
        if e != 0:
            raise CudaRtError("Stream destroy")

    def sync(self):
        """
        Synchronize on the cudastream owned by this object. Note that this could
        raise exception due to issues with previous asynchronous launches!
        """
        cdef _Error e = cudaStreamSynchronize(self.s)
        if e != 0:
            raise CudaRtError("Stream sync")

    cdef _Stream getStream(self):
        return self.s
