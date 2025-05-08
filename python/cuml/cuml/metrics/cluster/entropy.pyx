#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import math
import typing

import cupy as cp
import numpy as np
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.common import CumlArray
from cuml.internals.input_utils import input_to_cupy_array

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double entropy(const handle_t &handle,
                   const int *y,
                   const int n,
                   const int lower_class_range,
                   const int upper_class_range) except +


@cuml.internals.api_return_generic()
def _prepare_cluster_input(cluster) -> typing.Tuple[CumlArray, int, int, int]:
    """Helper function to avoid code duplication for clustering metrics."""
    cluster_m, n_rows, _, _ = input_to_cupy_array(
        cluster,
        check_dtype=np.int32,
        check_cols=1
    )

    lower_class_range = cp.min(cluster_m).item()
    upper_class_range = cp.max(cluster_m).item()

    return cluster_m, n_rows, lower_class_range, upper_class_range


@cuml.internals.api_return_any()
def cython_entropy(clustering, base=None, handle=None) -> float:
    """
    Computes the entropy of a distribution for given probability values.

    Parameters
    ----------
    clustering : array-like (device or host) shape = (n_samples,)
        Clustering of labels. Probabilities are computed based on occurrences
        of labels. For instance, to represent a fair coin (2 equally possible
        outcomes), the clustering could be [0,1]. For a biased coin with 2/3
        probability for tail, the clustering could be [0, 0, 1].
    base: float, optional
        The logarithmic base to use, defaults to e (natural logarithm).
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    S : float
        The calculated entropy.
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    (clustering, n_rows,
     lower_class_range, upper_class_range) = _prepare_cluster_input(clustering)

    cdef uintptr_t clustering_ptr = clustering.ptr

    S = entropy(handle_[0],
                <int*> clustering_ptr,
                <int> n_rows,
                <int> lower_class_range,
                <int> upper_class_range)

    if base is not None:
        # S needs to be converted from base e
        S = math.log(math.exp(S), base)

    return S
