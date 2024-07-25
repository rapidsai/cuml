#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_FULL_RO

from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from typing import Dict, List, Union
from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')

cdef extern from "treelite/c_api.h":
    cdef struct TreelitePyBufferFrame:
        void* buf
        char* format
        size_t itemsize
        size_t nitem

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass Model:
        vector[TreelitePyBufferFrame] SerializeToPyBuffer() except +
        @staticmethod
        unique_ptr[Model] DeserializeFromPyBuffer(const vector[TreelitePyBufferFrame] &) except +

cdef class PyBufferFrameWrapper:
    cdef TreelitePyBufferFrame _handle
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = self._handle.itemsize

        self.shape[0] = self._handle.nitem
        self.strides[0] = itemsize

        buffer.buf = self._handle.buf
        buffer.format = self._handle.format
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self._handle.nitem * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef PyBufferFrameWrapper MakePyBufferFrameWrapper(TreelitePyBufferFrame handle):
    cdef PyBufferFrameWrapper wrapper = PyBufferFrameWrapper()
    wrapper._handle = handle
    return wrapper

cdef list _get_frames(TreeliteModelHandle model):
    return [memoryview(MakePyBufferFrameWrapper(v))
            for v in (<Model*>model).SerializeToPyBuffer()]

cdef TreeliteModelHandle _init_from_frames(vector[TreelitePyBufferFrame] frames) except *:
    return <TreeliteModelHandle>Model.DeserializeFromPyBuffer(frames).release()


def get_frames(model: uintptr_t) -> List[memoryview]:
    return _get_frames(<TreeliteModelHandle> model)


def init_from_frames(frames: List[np.ndarray],
                     format_str: List[str], itemsize: List[int]) -> uintptr_t:
    cdef vector[TreelitePyBufferFrame] cpp_frames
    # Need to keep track of the buffers to release them later.
    cdef vector[Py_buffer] buffers
    cdef Py_buffer* buf
    cdef TreelitePyBufferFrame cpp_frame
    format_bytes = [s.encode('utf-8') for s in format_str]
    for i, frame in enumerate(frames):
        buffers.emplace_back()
        buf = &buffers.back()
        PyObject_GetBuffer(frame, buf, PyBUF_FULL_RO)
        cpp_frame.buf = buf.buf
        cpp_frame.format = format_bytes[i]
        cpp_frame.itemsize = itemsize[i]
        cpp_frame.nitem = buf.len // itemsize[i]
        cpp_frames.push_back(cpp_frame)
    output = <uintptr_t> _init_from_frames(cpp_frames)
    cdef int j
    for j in range(buffers.size()):
        PyBuffer_Release(&buffers[j])
    return output


def treelite_serialize(
    model: uintptr_t
) -> Dict[str, Union[List[str], List[np.ndarray]]]:
    frames = get_frames(model)
    header = {'format_str': [x.format for x in frames],
              'itemsize': [x.itemsize for x in frames]}
    return {'header': header, 'frames': [np.asarray(x) for x in frames]}


def treelite_deserialize(
    payload: Dict[str, Union[List[str], List[bytes]]]
) -> uintptr_t:
    header, frames = payload['header'], payload['frames']
    return init_from_frames(frames, header['format_str'], header['itemsize'])
