#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from rmm._lib.memory_resource cimport get_current_device_resource
from pylibraft.common.handle cimport handle_t
from cuml.manifold.umap_utils cimport *
from libcpp.utility cimport move
import numpy as np
import cupy as cp


cdef class GraphHolder:
    @staticmethod
    cdef GraphHolder new_graph(cuda_stream_view stream):
        cdef GraphHolder graph = GraphHolder.__new__(GraphHolder)
        graph.c_graph.reset(new COO(stream))
        graph.mr = get_current_device_resource()
        return graph

    @staticmethod
    cdef GraphHolder from_ptr(unique_ptr[COO]& ptr):
        cdef GraphHolder graph = GraphHolder.__new__(GraphHolder)
        graph.c_graph = move(ptr)
        return graph

    @staticmethod
    cdef GraphHolder from_coo_array(graph, handle, coo_array):
        def copy_from_array(dst_raft_coo_ptr, src_cp_coo):
            size = src_cp_coo.size
            itemsize = np.dtype(src_cp_coo.dtype).itemsize
            dest_buff = cp.cuda.UnownedMemory(ptr=dst_raft_coo_ptr,
                                              size=size * itemsize,
                                              owner=None,
                                              device_id=-1)
            dest_mptr = cp.cuda.memory.MemoryPointer(dest_buff, 0)
            src_buff = cp.cuda.UnownedMemory(ptr=src_cp_coo.data.ptr,
                                             size=size * itemsize,
                                             owner=None,
                                             device_id=-1)
            src_mptr = cp.cuda.memory.MemoryPointer(src_buff, 0)
            dest_mptr.copy_from_device(src_mptr, size * itemsize)

        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        graph.c_graph.reset(new COO(handle_.get_stream()))
        graph.get().allocate(coo_array.nnz,
                             coo_array.shape[0],
                             False,
                             handle_.get_stream())
        handle_.sync_stream()

        copy_from_array(graph.vals(), coo_array.data.astype('float32'))
        copy_from_array(graph.rows(), coo_array.row.astype('int32'))
        copy_from_array(graph.cols(), coo_array.col.astype('int32'))

        graph.mr = get_current_device_resource()
        return graph

    cdef inline COO* get(self):
        return self.c_graph.get()

    cdef uintptr_t vals(self):
        return <uintptr_t>self.get().vals()

    cdef uintptr_t rows(self):
        return <uintptr_t>self.get().rows()

    cdef uintptr_t cols(self):
        return <uintptr_t>self.get().cols()

    cdef uint64_t get_nnz(self):
        return self.get().nnz

    def get_cupy_coo(self):
        def create_nonowning_cp_array(ptr, dtype):
            mem = cp.cuda.UnownedMemory(ptr=ptr,
                                        size=(self.get_nnz() *
                                              np.dtype(dtype).itemsize),
                                        owner=self,
                                        device_id=-1)
            memptr = cp.cuda.memory.MemoryPointer(mem, 0)
            return cp.ndarray(self.get_nnz(), dtype=dtype, memptr=memptr)

        vals = create_nonowning_cp_array(self.vals(), np.float32)
        rows = create_nonowning_cp_array(self.rows(), np.int32)
        cols = create_nonowning_cp_array(self.cols(), np.int32)

        return cp.sparse.coo_matrix(((vals, (rows, cols))))

    def __dealloc__(self):
        self.c_graph.reset(NULL)


def find_ab_params(spread, min_dist):
    """ Function taken from UMAP-learn : https://github.com/lmcinnes/umap
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    from cuml.internals.import_utils import has_scipy
    if has_scipy():
        from scipy.optimize import curve_fit
    else:
        raise RuntimeError('Scipy is needed to run find_ab_params')

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
