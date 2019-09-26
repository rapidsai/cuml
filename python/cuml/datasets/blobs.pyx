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

    """
    Generator of datasets composed of isotropic Gaussian distributed clusters
    in GPU.


    Examples
    ---------

    .. code-block:: python

          from cuml import make_blobs

          data, labels = (n_samples=10, centers=3, n_features=2)

          print(data.copy_to_host())
          print(labels.copy_to_host())

    Output:

    .. code-block:: python

          [[-6.4611025   2.980582  ]
           [-1.8473494   6.4483595 ]
           [-0.48936838  5.255189  ]
           [-6.0078964   0.59910655]
           [-3.7753344   7.0041647 ]
           [-0.6350849   5.1219263 ]
           [-4.675709    3.0528255 ]
           [-5.933864    2.0036478 ]
           [-0.11404657  4.69242   ]
           [ 0.23619342  4.699105  ]]

          [0 2 1 0 2 1 0 0 1 1]

    Parameters
    -----------

    n_samples : int (default = 100)
        Total number of points equally divided among clusters. Alternatively,
        it is the total number of rows of the dataset and labels.
    n_features : int, optional (default=2)
        The number of features for each sample. Alternatively, the number of
        columns in the resulting dataset.
    centers : int or array-like (device or host) shape = (n_samples, n_features)  # noqa
        The number of centers to generate, or the fixed center locations.
        If centers is None, 3 centers are generated.
    cluster_std : float or array-like (device or host) (default = 1.0)
        The standard deviation of the clusters.
    center_box : tuple of floats (min, max), optional (default = (-10.0, 10.0))
        The bounding box for cluster centers when generated at random.
    shuffle : boolean, optional (default=True)
        Whether to shuffle the samples.
    random_state : int, RandomState instance or None (default)
        Seed for the random number generator for dataset creation
    handle : cuml.Handle
        If it is None, a new one is created just for this class

    """

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
                   <bool> shuffle,
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
                   <bool> shuffle,
                   <double> center_box_min,
                   <double> center_box_max,
                   <uint64_t> random_state)

    return out, labels
