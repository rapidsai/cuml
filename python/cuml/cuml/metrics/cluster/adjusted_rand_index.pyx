#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import cupy as cp
import warnings

from libc.stdint cimport uintptr_t

import cuml.internals
from raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from raft.common.handle import Handle
cimport cuml.common.cuda

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":

    double adjusted_rand_index(handle_t &handle,
                               int *y,
                               int *y_hat,
                               int n)


@cuml.internals.api_return_any()
def adjusted_rand_score(labels_true, labels_pred, handle=None,
                        convert_dtype=True) -> float:
    """
    Adjusted_rand_score is a clustering similarity metric based on the Rand
    index and is corrected for chance.

    Parameters
    ----------
        labels_true : Ground truth labels to be used as a reference

        labels_pred : Array of predicted labels used to evaluate the model

        handle : cuml.Handle

    Returns
    -------
        float
            The adjusted rand index value between -1.0 and 1.0
    """
    handle = Handle() \
        if handle is None else handle
    cdef handle_t* handle_ =\
        <handle_t*><size_t>handle.getHandle()

    labels_true, n_rows, _, _ = \
        input_to_cuml_array(labels_true, order='C', check_dtype=cp.int32,
                            convert_to_dtype=(cp.int32 if convert_dtype
                                              else None))

    labels_pred, _, _, _ = \
        input_to_cuml_array(labels_pred, order='C', check_dtype=cp.int32,
                            convert_to_dtype=(cp.int32 if convert_dtype
                                              else None))

    rand_score = adjusted_rand_index(handle_[0],
                                     <int*><uintptr_t> labels_true.ptr,
                                     <int*><uintptr_t> labels_pred.ptr,
                                     <int> n_rows)

    return rand_score
