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

import numpy as np

from libc.stdint cimport uintptr_t

from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_dev_array
import cuml.common.handle
cimport cuml.common.cuda

cdef extern from "metrics/metrics.hpp" namespace "ML::Metrics":

    float accuracy_score_py(cumlHandle &handle,
                            int *predictions,
                            int *ref_predictions,
                            int n)


def accuracy_score(predictions, labels, handle=None):

    handle = cuml.common.handle.Handle() \
        if handle is None else handle
    cdef cumlHandle* handle_ =\
        <cumlHandle*><size_t>handle.getHandle()

    cdef uintptr_t preds_ptr, labels_ptr
    preds_m, preds_ptr, n_rows, _, _ = \
        input_to_dev_array(predictions)

    labels_m, labels_ptr, _, _, labels_dtype = \
        input_to_dev_array(labels)

    acc = accuracy_score_py(handle_[0],
                            <int*> preds_ptr,
                            <int*> labels_ptr,
                            <int> n_rows)

    return acc
