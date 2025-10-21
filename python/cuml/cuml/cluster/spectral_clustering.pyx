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

import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import scipy.sparse as sp
from pylibraft.common.handle import Handle

import cuml
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.utils import check_random_seed

from libc.stdint cimport int64_t, uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/datasets/make_blobs.hpp" namespace "ML::Datasets" nogil:
    cdef void make_blobs(const handle_t& handle,
                         float* out,
                         int64_t* labels,
                         int64_t n_rows,
                         int64_t n_cols,
                         int64_t n_clusters,
                         bool row_major,
                         const float* centers,
                         const float* cluster_std,
                         const float cluster_std_scalar,
                         bool shuffle,
                         float center_box_min,
                         float center_box_max,
                         uint64_t seed) except +


cdef extern from "cuml/cluster/spectral_clustering.hpp" \
        namespace "ML::SpectralClustering" nogil:

    cdef cppclass SpectralClusteringParams:
        int n_clusters
        int n_components
        int n_neighbors
        int n_init
        uint64_t seed

    cdef void fit_predict(
        const handle_t &handle,
        const SpectralClusteringParams &config,
        const int* coo_rows,
        const int* coo_cols,
        const float* coo_vals,
        int nnz,
        int n_rows,
        int n_cols,
        int* labels) except +


@cuml.internals.api_return_array(get_output_type=True)
def spectral_clustering(A,
                        *,
                        int n_clusters=8,
                        random_state=None,
                        n_components=None,
                        n_neighbors=10,
                        n_init=10,
                        handle=None):
    if handle is None:
        handle = Handle()

    # Convert to COO sparse matrix
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

    # NOTE: C++ gtest includes diagonal (self-loops) in the connectivity graph
    # so we should NOT remove them to match the C++ behavior
    # # Remove diagonal
    valid = affinity_rows != affinity_cols
    if not valid.all():
        affinity_data = affinity_data[valid]
        affinity_rows = affinity_rows[valid]
        affinity_cols = affinity_cols[valid]
        affinity_nnz = len(affinity_data)

    cdef float* affinity_data_ptr = <float*><uintptr_t>affinity_data.data.ptr
    cdef int* affinity_rows_ptr = <int*><uintptr_t>affinity_rows.data.ptr
    cdef int* affinity_cols_ptr = <int*><uintptr_t>affinity_cols.data.ptr

    cdef int n_samples = A.shape[0]
    cdef int nnz = affinity_nnz

    # Allocate labels
    labels = CumlArray.empty(n_samples, dtype=np.int32)
    cdef int* labels_ptr = <int*><uintptr_t>labels.ptr

    cdef SpectralClusteringParams config
    config.n_clusters = n_clusters
    config.n_components = n_components if n_components is not None else n_clusters
    config.n_neighbors = n_neighbors
    config.n_init = n_init
    config.seed = check_random_seed(random_state)

    cdef handle_t *handle_ = <handle_t*><size_t>handle.getHandle()

    with nogil:
        fit_predict(
            handle_[0],
            config,
            affinity_rows_ptr,
            affinity_cols_ptr,
            affinity_data_ptr,
            nnz,
            n_samples,
            n_samples,
            labels_ptr
        )

    return labels


class SpectralClustering(Base):
    def __init__(self, n_clusters=8, n_components=None, random_state=None,
                 n_neighbors=10, n_init=10, affinity='precomputed',
                 handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.affinity = affinity

    def fit_predict(self, X, y=None) -> CumlArray:
        self.labels_ = spectral_clustering(
            X,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            n_init=self.n_init,
            handle=self.handle
        )
        return self.labels_

    def fit(self, X, y=None) -> "SpectralClustering":
        self.fit_predict(X, y)
        return self


def raft_make_blobs(n_samples, n_features, n_clusters,
                    cluster_std=1.0, shuffle=True,
                    center_box_min=-10.0, center_box_max=10.0,
                    random_state=None, handle=None):
    """
    Wrapper for raft's C++ make_blobs implementation.
    This ensures we use the same data generation as the C++ tests.
    """
    if handle is None:
        handle = Handle()

    cdef handle_t *handle_ = <handle_t*><size_t>handle.getHandle()

    # Allocate output arrays
    X = cp.empty((n_samples, n_features), dtype=np.float32)
    y = cp.empty(n_samples, dtype=np.int64)

    # Get pointers
    cdef uintptr_t X_ptr = X.data.ptr
    cdef uintptr_t y_ptr = y.data.ptr

    # Set random seed
    if random_state is None:
        random_state = 0
    cdef uint64_t seed = check_random_seed(random_state)

    # Cast Python variables to C types before nogil block
    cdef int64_t c_n_samples = n_samples
    cdef int64_t c_n_features = n_features
    cdef int64_t c_n_clusters = n_clusters
    cdef float c_cluster_std = cluster_std
    cdef bool c_shuffle = shuffle
    cdef float c_center_box_min = center_box_min
    cdef float c_center_box_max = center_box_max

    # Call C++ make_blobs
    with nogil:
        make_blobs(handle_[0],
                   <float*>X_ptr,
                   <int64_t*>y_ptr,
                   c_n_samples,
                   c_n_features,
                   c_n_clusters,
                   True,  # row_major
                   NULL,  # centers
                   NULL,  # cluster_std array
                   c_cluster_std,
                   c_shuffle,
                   c_center_box_min,
                   c_center_box_max,
                   seed)

    return X, y
