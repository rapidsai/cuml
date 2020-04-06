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
import warnings

from libc.stdint cimport uintptr_t

from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_dev_array
import cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":

    double adjustedRandIndex(cumlHandle &handle,
                             int *y,
                             int *y_hat,
                             int n,
                             int lower_class_range,
                             int upper_class_range)


def adjusted_rand_score(labels_true, labels_pred, handle=None):
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
    handle = cuml.common.handle.Handle() \
        if handle is None else handle
    cdef cumlHandle* handle_ =\
        <cumlHandle*><size_t>handle.getHandle()
    if labels_true.astype != np.int64:
        warnings.warn(" The dtype of ground truth is not int32"
                      " converting the ground truth to int32")
        labels_true = labels_true.astype(np.int32)
    if labels_pred.astype != np.int32:
        warnings.warn(" The dtype of predicted labels is not int32"
                      " converting the predicted labels to int32")
        labels_pred = labels_pred.astype(np.int32)

    min_val_y = np.nanmin(labels_true)
    lower_class_range = np.nanmin(labels_pred) \
        if min_val_y > np.nanmin(labels_pred) else np.nanmin(labels_true)
    max_val_y = np.nanmax(labels_true)
    upper_class_range = np.nanmax(labels_pred) \
        if max_val_y < np.nanmax(labels_pred) else np.nanmax(labels_true)
    cdef uintptr_t y_ptr, y_hat_ptr
    y_m, y_ptr, n_rows, _, _ = \
        input_to_dev_array(labels_true)

    y_hat_m, y_hat_ptr, _, _, y_hat_dtype = \
        input_to_dev_array(labels_pred)

    rand_score = adjustedRandIndex(handle_[0],
                                   <int*> y_ptr,
                                   <int*> y_hat_ptr,
                                   <int> n_rows,
                                   <int> lower_class_range,
                                   <int> upper_class_range)

    return rand_score
