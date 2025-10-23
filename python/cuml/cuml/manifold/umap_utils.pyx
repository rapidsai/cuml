#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import ctypes
from typing import Literal

import cupy as cp
import cupyx
import numpy as np
import scipy

from libcpp.utility cimport move
from pylibraft.common.handle cimport handle_t
from rmm.pylibrmm.memory_resource cimport get_current_device_resource

from cuml.manifold.umap_utils cimport *
from cuml.metrics.distance_type cimport DistanceType


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
    cdef GraphHolder from_coo_array(handle, coo_array):
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

        cdef GraphHolder graph = GraphHolder.__new__(GraphHolder)
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

    cdef inline COO* get(self) noexcept:
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

        return cupyx.scipy.sparse.coo_matrix(((vals, (rows, cols))))

    def __dealloc__(self):
        self.c_graph.reset(NULL)

cdef class HostGraphHolder:
    @staticmethod
    cdef HostGraphHolder new_graph():
        cdef HostGraphHolder graph = HostGraphHolder.__new__(HostGraphHolder)
        graph.c_graph.reset(new host_COO())
        return graph

    cdef uintptr_t vals(self):
        return <uintptr_t>self.get().vals()

    cdef uintptr_t rows(self):
        return <uintptr_t>self.get().rows()

    cdef uintptr_t cols(self):
        return <uintptr_t>self.get().cols()

    cdef uint64_t get_nnz(self):
        return self.get().get_nnz()

    def get_scipy_coo(self):
        """Convert the host graph to a SciPy COO sparse matrix.

        Returns
        -------
        scipy.sparse.coo_matrix
            A copy of the graph data as a SciPy COO sparse matrix.
            Note that this returns a copy of the data, not a view.
        """
        def create_nonowning_numpy_array(ptr, dtype):
            c_type = np.ctypeslib.as_ctypes_type(dtype)
            c_pointer = ctypes.cast(ptr, ctypes.POINTER(c_type))
            return np.ctypeslib.as_array(c_pointer, shape=(self.get_nnz(),))

        vals = create_nonowning_numpy_array(self.vals(), np.float32)
        rows = create_nonowning_numpy_array(self.rows(), np.int32)
        cols = create_nonowning_numpy_array(self.cols(), np.int32)

        graph = scipy.sparse.coo_matrix((vals.copy(), (rows.copy(), cols.copy())))
        return graph

    cdef inline host_COO* get(self) noexcept:
        return self.c_graph.get()

    cdef inline cppHostCOO* ref(self) noexcept:
        return <cppHostCOO*>self.c_graph.get()

    def __dealloc__(self):
        self.c_graph.reset(NULL)


def find_ab_params(spread, min_dist):
    """ Function taken from UMAP-learn : https://github.com/lmcinnes/umap
    Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


_METRICS = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
    "cityblock": DistanceType.L1,
    "l1": DistanceType.L1,
    "manhattan": DistanceType.L1,
    "taxicab": DistanceType.L1,
    "minkowski": DistanceType.LpUnexpanded,
    "chebyshev": DistanceType.Linf,
    "linf": DistanceType.Linf,
    "cosine": DistanceType.CosineExpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "hamming": DistanceType.HammingUnexpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "canberra": DistanceType.Canberra
}

_SUPPORTED_METRICS = {
    "nn_descent": {
        "sparse": frozenset(),
        "dense": frozenset((
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.CosineExpanded,
        ))
    },
    "brute_force_knn": {
        "sparse": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            DistanceType.JaccardExpanded,
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        )),
        "dense": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            # DistanceType.JaccardExpanded,  # not supported
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        ))
    }
}


def coerce_metric(
    metric: str,
    sparse: bool = False,
    build_algo: Literal["brute_force_knn", "nn_descent"] = "brute_force_knn",
) -> DistanceType:
    """Coerce a metric string to a `DistanceType`.

    Also checks that the metric is valid and supported.
    """
    if not isinstance(metric, str):
        raise TypeError(f"Expected `metric` to be a str, got {type(metric).__name__}")

    try:
        out = _METRICS[metric.lower()]
    except KeyError:
        raise ValueError(f"Invalid value for metric: {metric!r}")

    kind = "sparse" if sparse else "dense"
    supported = _SUPPORTED_METRICS[build_algo][kind]
    if out not in supported:
        raise NotImplementedError(
            f"Metric {metric!r} not supported for {kind} inputs with {build_algo=}"
        )

    return out
