#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import math

import numpy as np
import cupy as cp

from libc.stdint cimport uintptr_t

from cuml.common.handle cimport cumlHandle
from cuml.utils import with_cupy_rmm, input_to_cuml_array
import cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double entropy(const cumlHandle &handle,
                   const int *y,
                   const int n,
                   const int lower_class_range,
                   const int upper_class_range) except +


@with_cupy_rmm
def prepare_data(labels_true, labels_pred=None):
    """Helper function to avoid code duplication for clustering metrics."""
    ground_truth_m, n_rows, _, _ = input_to_cuml_array(
        labels_true,
        check_dtype=np.int32,
        check_cols=1
    )

    if labels_pred is not None:
        preds_m, n_rows, _, _ = input_to_cuml_array(
            labels_pred,
            check_dtype=np.int32,
            check_rows=n_rows,
            check_cols=1
        )
        cp_preds_m = preds_m.to_output(output_type='cupy')
    else:
        preds_m = None
        cp_preds_m = cp.empty((0,))

    cp_ground_truth_m = ground_truth_m.to_output(output_type='cupy')


    lower_class_range = min(cp.min(cp_ground_truth_m),
                            cp.min(cp_preds_m))
    upper_class_range = max(cp.max(cp_ground_truth_m),
                            cp.max(cp_preds_m))

    return (ground_truth_m, preds_m,
            n_rows,
            lower_class_range, upper_class_range)


def cython_entropy(pk, base=None, handle=None):
    """
    Computes the entropy of a distribution for given probability values.

    Parameters
    ----------
    pk : array-like (device or host) shape = (n_samples,)
        Defines the (discrete) distribution. pk[i] is the unnormalized
        probability of event i.
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
    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle *handle_ = <cumlHandle*> <size_t> handle.getHandle()

    (pk_ary, _,
     n_rows,
     lower_class_range, upper_class_range) = prepare_data(pk)

    cdef uintptr_t pk_ptr = pk_ary.ptr

    S = entropy(handle_[0],
                <int*> pk_ptr,
                <int> n_rows,
                <int> lower_class_range,
                <int> upper_class_range)

    if base is not None:
        # S needs to be converted from base e
        S = math.log(math.exp(S), base)

    return S
