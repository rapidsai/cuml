#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import numpy as np

from cuml.internals import get_handle
from cuml.internals.input_utils import input_to_cupy_array

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double entropy(const handle_t &handle,
                   const int *y,
                   const int n,
                   const int lower_class_range,
                   const int upper_class_range) except +


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
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    Returns
    -------
    S : float
        The calculated entropy.
    """
    handle = get_handle(handle=handle)
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    clustering, n_rows, _, _ = input_to_cupy_array(
        clustering,
        check_dtype=np.int32,
        check_cols=1
    )
    lower_class_range = cp.min(clustering).item()
    upper_class_range = cp.max(clustering).item()

    cdef uintptr_t clustering_ptr = clustering.data.ptr

    S = entropy(handle_[0],
                <int*> clustering_ptr,
                <int> n_rows,
                <int> lower_class_range,
                <int> upper_class_range)

    if base is not None:
        # S needs to be converted from base e
        S = math.log(math.exp(S), base)

    return S
