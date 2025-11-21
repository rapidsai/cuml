#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import scipy.sparse as sp
from pylibraft.common.handle import Handle

import cuml
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.utils import check_random_seed

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources


cdef extern from "cuml/cluster/spectral_clustering.hpp" \
        namespace "ML::SpectralClustering" nogil:

    cdef cppclass params:
        int n_clusters
        int n_components
        int n_neighbors
        int n_init
        float eigen_tol
        uint64_t seed

    cdef void fit_predict(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_vector_view[int, int] labels) except +

    cdef void fit_predict(
        const device_resources &handle,
        params config,
        device_vector_view[int, int] rows,
        device_vector_view[int, int] cols,
        device_vector_view[float, int] vals,
        device_vector_view[int, int] labels) except +


@cuml.internals.api_return_array(get_output_type=True)
def spectral_clustering(A,
                        *,
                        int n_clusters=8,
                        random_state=None,
                        n_components=None,
                        n_neighbors=10,
                        n_init=10,
                        eigen_tol=0.0,
                        affinity='precomputed',
                        handle=None):
    if handle is None:
        handle = Handle()

    cdef float* dataset_ptr = NULL
    cdef float* affinity_data_ptr = NULL
    cdef int* affinity_rows_ptr = NULL
    cdef int* affinity_cols_ptr = NULL
    cdef int n_samples, n_features
    cdef int nnz = 0
    cdef bool is_precomputed = (affinity == 'precomputed')

    if not is_precomputed:
        # Input is a dataset, convert to appropriate format
        from cuml.internals.input_utils import input_to_cupy_array
        A = input_to_cupy_array(
            A, order="C", check_dtype=np.float32, convert_to_dtype=cp.float32
        ).array
        n_samples, n_features = A.shape
        dataset_ptr = <float*><uintptr_t>A.data.ptr
    elif cp_sp.issparse(A):
        A = A.tocoo()
        if A.dtype != np.float32:
            A = A.astype("float32")
    elif sp.issparse(A):
        A = cp_sp.coo_matrix(A, dtype="float32")
    else:
        A = cp_sp.coo_matrix(cp.asarray(A, dtype="float32"))

    if is_precomputed:
        A.sum_duplicates()

        affinity_data = A.data
        affinity_rows = A.row.astype(np.int32)
        affinity_cols = A.col.astype(np.int32)
        affinity_nnz = A.nnz

        # NOTE: C++ gtest includes diagonal (self-loops) in the connectivity graph
        # so we should NOT remove them to match the C++ behavior
        # # Remove diagonal
        valid = affinity_rows != affinity_cols
        if not valid.all():
            affinity_data = affinity_data[valid]
            affinity_rows = affinity_rows[valid]
            affinity_cols = affinity_cols[valid]
            affinity_nnz = len(affinity_data)

        affinity_data_ptr = <float*><uintptr_t>affinity_data.data.ptr
        affinity_rows_ptr = <int*><uintptr_t>affinity_rows.data.ptr
        affinity_cols_ptr = <int*><uintptr_t>affinity_cols.data.ptr

        n_samples = A.shape[0]
        nnz = affinity_nnz

    # Allocate labels
    labels = CumlArray.empty(n_samples, dtype=np.int32)
    cdef int* labels_ptr = <int*><uintptr_t>labels.ptr

    cdef params config
    config.n_clusters = n_clusters
    config.n_components = n_components if n_components is not None else n_clusters
    config.n_neighbors = n_neighbors
    config.n_init = n_init
    config.eigen_tol = eigen_tol
    config.seed = check_random_seed(random_state)

    cdef device_resources *handle_ = <device_resources*><size_t>handle.getHandle()

    with nogil:
        if is_precomputed:
            fit_predict(
                handle_[0],
                config,
                make_device_vector_view(affinity_rows_ptr, nnz),
                make_device_vector_view(affinity_cols_ptr, nnz),
                make_device_vector_view(affinity_data_ptr, nnz),
                make_device_vector_view(labels_ptr, n_samples)
            )
        else:
            fit_predict(
                handle_[0],
                config,
                make_device_matrix_view[float, int, row_major](
                    dataset_ptr, n_samples, n_features
                ),
                make_device_vector_view(labels_ptr, n_samples)
            )

    return labels


class SpectralClustering(Base):
    def __init__(self, n_clusters=8, n_components=None, random_state=None,
                 n_neighbors=10, n_init=10, eigen_tol=0.0, affinity='precomputed',
                 handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.eigen_tol = eigen_tol
        self.affinity = affinity

    def fit_predict(self, X, y=None) -> CumlArray:
        self.labels_ = spectral_clustering(
            X,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            n_init=self.n_init,
            eigen_tol=self.eigen_tol,
            affinity=self.affinity,
            handle=self.handle
        )
        return self.labels_

    def fit(self, X, y=None) -> "SpectralClustering":
        self.fit_predict(X, y)
        return self
