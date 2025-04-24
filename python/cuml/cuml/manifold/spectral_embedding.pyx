#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint32_t, uint64_t, uintptr_t

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.random.cpp.rng_state cimport RngState


cdef extern from "cuml/manifold/spectral_embedding_types.hpp" namespace "ML":
    cdef cppclass spectral_embedding_config:
        int n_components
        bool norm_laplacian
        bool drop_first
        uint64_t seed

cdef spectral_embedding_config config

cdef extern from "cuml/manifold/spectral_embedding.hpp":

    cdef int spectral_embedding(
        const device_resources &handle,
        device_matrix_view[float, int, row_major] nums,
        spectral_embedding_config config) except +

@auto_sync_handle
def get_affinity_matrix(A, handle=None):

    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    print(A)
    A = cai_wrapper(A)
    A_ptr = <uintptr_t>A.data

    print(A)
    print(A.shape)

    config.n_components = 2
    config.norm_laplacian = True
    config.drop_first = True
    config.seed = 1234

    # X_m = SparseCumlArray(A, convert_to_dtype=cp.float32,
    #                         convert_format=False)

    # # Need to establish result matrices for indices (Nxk)
    # # and for distances (Nxk)
    # I_ndarr = CumlArray.zeros((X_m.shape[0], n_neighbors),
    #                             dtype=np.int32, order="C")
    # D_ndarr = CumlArray.zeros((X_m.shape[0], n_neighbors),
    #                             dtype=np.float32, order="C")

    # cdef uintptr_t _I_ptr = I_ndarr.ptr
    # cdef uintptr_t _D_ptr = D_ndarr.ptr

    # eigenvectors_cai = cai_wrapper(eigenvectors)
    # eigenvectors_ptr = <uintptr_t>eigenvectors_cai.data

    # make_device_matrix_view[float, uint32_t, col_major](
    #             <float *>eigenvectors_ptr, <uint32_t> N, <uint32_t> k)

    cdef int result = spectral_embedding(deref(h), make_device_matrix_view[float, int, row_major](<float *>A_ptr, <int> A.shape[0], <int> A.shape[1]), config)

    return result
