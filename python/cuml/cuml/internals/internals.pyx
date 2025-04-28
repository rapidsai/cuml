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

from numba.cuda.api import from_cuda_array_interface


cdef extern from "Python.h":
    cdef cppclass PyObject


from libc.stdint cimport uintptr_t


cdef extern from "callbacks_implems.h" namespace "ML::Internals":
    cdef cppclass Callback:
        pass

    cdef cppclass DefaultGraphBasedDimRedCallback(Callback):
        void setup(int n, int d) except +
        void on_preprocess_end(void* embeddings) except +
        void on_epoch_end(void* embeddings) except +
        void on_train_end(void* embeddings) except +
        PyObject* pyCallbackClass

cdef class PyCallback:

    def get_numba_matrix(self, embeddings, shape, typestr):

        sizeofType = 4 if typestr == "float32" else 8
        desc = {
            'shape': shape,
            'strides': (shape[1]*sizeofType, sizeofType),
            'typestr': typestr,
            'data': [embeddings],
            'order': 'C',
            'version': 1
        }

        return from_cuda_array_interface(desc)

cdef class GraphBasedDimRedCallback(PyCallback):
    """
    Usage
    -----

    class CustomCallback(GraphBasedDimRedCallback):
        def on_preprocess_end(self, embeddings):
            print(embeddings.copy_to_host())

        def on_epoch_end(self, embeddings):
            print(embeddings.copy_to_host())

        def on_train_end(self, embeddings):
            print(embeddings.copy_to_host())

    reducer = UMAP(n_components=2, callback=CustomCallback())
    """

    cdef DefaultGraphBasedDimRedCallback native_callback

    def __cinit__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def __reduce__(self):
        return (type(self), ())

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)

    def on_preprocess_end(self, embeddings):
        pass

    def on_epoch_end(self, embeddings):
        pass

    def on_train_end(self, embeddings):
        pass
