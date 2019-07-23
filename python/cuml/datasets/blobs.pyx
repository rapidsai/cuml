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

import cuml
import numpy as np

from collections.abc import Sequence

from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, zeros, input_to_dev_array

from libcpp cimport bool
from libc.stdint cimport uint64_t, uintptr_t

from numba import cuda

from random import randint

cdef extern from "datasets/make_blobs.hpp" namespace "ML::Datasets":
    cdef void make_blobs(const cumlHandle& handle,
                         float* out,
                         int* labels,
                         int n_rows,
                         int n_cols,
                         int n_clusters,
                         const float* centers,
                         const float* cluster_std,
                         const float cluster_std_scalar,
                         bool shuffle,
                         float center_box_min,
                         float center_box_max,
                         uint64_t seed) except +

    cdef void make_blobs(cumlHandle& handle,
                         double* out,
                         int* labels,
                         int n_rows,
                         int n_cols,
                         int n_clusters,
                         double* centers,
                         double* cluster_std,
                         double cluster_std_scalar,
                         bool shuffle,
                         double center_box_min,
                         double center_box_max,
                         uint64_t seed) except +

inp_to_dtype = {
    'single': np.float32,
    'float': np.float32,
    'double': np.float64,
    np.float32: np.float32,
    np.float64: np.float64
}


# Note: named blobs to avoid cython naming conflict issues, renaming in
# __init__.py to make_blob
def blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,
          center_box=(-10.0, 10.0), shuffle=True, random_state=None,
          dtype='single', handle=None):

    if dtype not in ['single', 'float', 'double', np.float32, np.float64]:
        raise TypeError("dtype must be either 'float' or 'double'")
    else:
        dtype = inp_to_dtype[dtype]

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    out = zeros((n_samples, n_features), dtype=dtype, order='C')
    cdef uintptr_t out_ptr = get_dev_array_ptr(out)

    labels = zeros(n_samples, dtype=np.int32)
    cdef uintptr_t labels_ptr = get_dev_array_ptr(labels)

    cdef uintptr_t centers_ptr
    centers_ptr = <uintptr_t> NULL

    if centers is not None:
        if isinstance(centers, int):
            n_clusters = centers
            n_rows_centers = 1

        else:
            centers, centers_ptr, n_rows_centers, _, _ = \
                input_to_dev_array(centers, convert_to_dtype=dtype,
                                   check_cols=n_features)

            n_clusters = len(centers)

    else:
        n_clusters = 3
        n_rows_centers = 1

    cdef uintptr_t cluster_std_ptr

    if isinstance(cluster_std, float):
        cluster_std_ptr = <uintptr_t> NULL

    else:
        cluster_std_ary, cluster_std_ptr, _, _, _ = \
            input_to_dev_array(cluster_std, convert_to_dtype=dtype,
                               check_cols=n_features,
                               check_rows=n_rows_centers)
        cluster_std = -1.0

    center_box_min = center_box[0]
    center_box_max = center_box[1]

    if random_state is None:
        random_state = randint(0, 1e18)

    if dtype == np.float32:
        make_blobs(handle_[0],
                   <float*> out_ptr,
                   <int*> labels_ptr,
                   <int> n_samples,
                   <int> n_features,
                   <int> n_clusters,
                   <float*> centers_ptr,
                   <float*> cluster_std_ptr,
                   <float> cluster_std,
                   <bool> True,
                   <float> center_box_min,
                   <float> center_box_max,
                   <uint64_t> random_state)

    else:
        make_blobs(handle_[0],
                   <double*> out_ptr,
                   <int*> labels_ptr,
                   <int> n_samples,
                   <int> n_features,
                   <int> n_clusters,
                   <double*> centers_ptr,
                   <double*> cluster_std_ptr,
                   <double> cluster_std,
                   <bool> True,
                   <double> center_box_min,
                   <double> center_box_max,
                   <uint64_t> random_state)

    return out, labels
