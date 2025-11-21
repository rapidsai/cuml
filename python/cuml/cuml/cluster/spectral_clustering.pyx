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
from cuml.common.array_descriptor import CumlArrayDescriptor
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

    cdef float* affinity_data_ptr = NULL
    cdef int* affinity_rows_ptr = NULL
    cdef int* affinity_cols_ptr = NULL
    cdef int affinity_nnz = 0

    if affinity == "nearest_neighbors":
        from cuml.internals.input_utils import input_to_cupy_array
        A = input_to_cupy_array(
            A, order="C", check_dtype=np.float32, convert_to_dtype=cp.float32
        ).array

        affinity_data_ptr = <float*><uintptr_t>A.data.ptr

        isfinite = cp.isfinite(A).all()
    elif affinity == "precomputed":
        # Coerce `A` to a canonical float32 COO sparse matrix
        if cp_sp.issparse(A):
            A = A.tocoo()
            if A.dtype != np.float32:
                A = A.astype("float32")
        elif sp.issparse(A):
            A = cp_sp.coo_matrix(A, dtype="float32")
        else:
            A = cp_sp.coo_matrix(cp.asarray(A, dtype="float32"))
        A.sum_duplicates()

        affinity_data = A.data
        affinity_rows = A.row.astype(np.int32)
        affinity_cols = A.col.astype(np.int32)
        affinity_nnz = A.nnz

        # Remove diagonal elements
        valid = affinity_rows != affinity_cols
        if not valid.all():
            affinity_data = affinity_data[valid]
            affinity_rows = affinity_rows[valid]
            affinity_cols = affinity_cols[valid]
            affinity_nnz = len(affinity_data)

        affinity_data_ptr = <float*><uintptr_t>affinity_data.data.ptr
        affinity_rows_ptr = <int*><uintptr_t>affinity_rows.data.ptr
        affinity_cols_ptr = <int*><uintptr_t>affinity_cols.data.ptr

        isfinite = cp.isfinite(affinity_data).all()
    else:
        raise ValueError(
            f"`affinity={affinity!r}` is not supported, expected one of "
            "['nearest_neighbors', 'precomputed']"
        )

    cdef int n_samples, n_features
    n_samples, n_features = A.shape

    if not isfinite:
        raise ValueError(
            "Input contains NaN or inf; nonfinite values are not supported"
        )

    if n_samples < 2:
        raise ValueError(
            f"Found array with {n_samples} sample(s) (shape={A.shape}) while a "
            f"minimum of 2 is required."
        )

    # Allocate output array
    labels = CumlArray.empty(n_samples, dtype=np.int32)

    cdef params config
    config.seed = check_random_seed(random_state)
    config.n_clusters = n_clusters
    config.n_components = n_components if n_components is not None else n_clusters
    config.n_neighbors = n_neighbors
    config.n_init = n_init
    config.eigen_tol = eigen_tol

    cdef int* labels_ptr = <int*><uintptr_t>labels.ptr
    cdef bool precomputed = affinity == "precomputed"
    cdef device_resources *handle_ = <device_resources*><size_t>handle.getHandle()

    with nogil:
        if precomputed:
            fit_predict(
                handle_[0],
                config,
                make_device_vector_view(affinity_rows_ptr, affinity_nnz),
                make_device_vector_view(affinity_cols_ptr, affinity_nnz),
                make_device_vector_view(affinity_data_ptr, affinity_nnz),
                make_device_vector_view(labels_ptr, n_samples)
            )
        else:
            fit_predict(
                handle_[0],
                config,
                make_device_matrix_view[float, int, row_major](
                    affinity_data_ptr, n_samples, n_features
                ),
                make_device_vector_view(labels_ptr, n_samples)
            )

    return labels


class SpectralClustering(Base):
    labels_ = CumlArrayDescriptor()

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
