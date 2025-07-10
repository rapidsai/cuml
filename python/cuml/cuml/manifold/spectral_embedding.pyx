#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
from libc.stdint cimport uint64_t, uintptr_t

from pylibraft.common import cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    make_device_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources

from cuml.common import input_to_cuml_array
from cuml.internals.base import Base
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin


cdef extern from "cuvs/preprocessing/spectral_embedding.hpp" namespace \
        "cuvs::preprocessing::spectral_embedding" nogil:
    cdef cppclass params:
        int n_components
        int n_neighbors
        bool norm_laplacian
        bool drop_first
        uint64_t seed

cdef params config

cdef extern from "cuml/manifold/spectral_embedding.hpp" namespace "ML::SpectralEmbedding":

    cdef int spectral_embedding_cuvs(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_matrix_view[float, int, col_major] embedding) except +


@auto_sync_handle
def spectral_embedding(A,
                       n_components,
                       random_state=None,
                       n_neighbors=None,
                       norm_laplacian=True,
                       drop_first=True,
                       handle=None):

    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    A, _n_rows, _n_cols, _ = \
        input_to_cuml_array(A, order="C", check_dtype=np.float32,
                            convert_to_dtype=cp.float32)
    A_ptr = <uintptr_t>A.ptr

    config.n_components = n_components
    config.seed = random_state if random_state is not None else 42

    config.n_neighbors = (
        n_neighbors
        if n_neighbors is not None
        else max(int(A.shape[0] / 10), 1)
    )

    config.norm_laplacian = norm_laplacian
    config.drop_first = drop_first

    if config.drop_first:
        config.n_components += 1

    eigenvectors = device_ndarray.empty((A.shape[0], n_components), dtype=A.dtype, order='F')

    eigenvectors_cai = cai_wrapper(eigenvectors)
    eigenvectors_ptr = <uintptr_t>eigenvectors_cai.data

    cdef int _result = spectral_embedding_cuvs(
        deref(h), config,
        make_device_matrix_view[float, int, row_major](
            <float *>A_ptr, <int> A.shape[0], <int> A.shape[1]),
        make_device_matrix_view[float, int, col_major](
            <float *>eigenvectors_ptr, <int> A.shape[0], <int> n_components))

    return cp.asarray(eigenvectors)


class SpectralEmbedding(Base,
                        CMajorInputTagMixin,
                        SparseInputTagMixin):

    def __init__(self, n_components=2, random_state=None, n_neighbors=None,
                 handle=None):
        super().__init__(handle=handle)
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    def fit(self, X, y=None):
        self.embedding_ = self._fit(X, self.n_components,
                                    random_state=self.random_state,
                                    n_neighbors=self.n_neighbors)
        return self

    def _fit(self, A, n_components, random_state=None, n_neighbors=None):
        return spectral_embedding(A, n_components,
                                  random_state=random_state,
                                  n_neighbors=n_neighbors)
