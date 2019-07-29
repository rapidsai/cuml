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


def accuracy_score(ground_truth, predictions, handle=None):
    """
    Calcuates the accuracy score of a classification model.

        Parameters
        ----------
            handle : cuml.Handle
            prediction : The lables predicted by the model
                         for the test dataset
            ground_truth : The ground truth labels of the test dataset
        Returns
        -------
            The accuracy of the model used for prediction
    """
    handle = cuml.common.handle.Handle() \
        if handle is None else handle
    cdef cumlHandle* handle_ =\
        <cumlHandle*><size_t>handle.getHandle()

    cdef uintptr_t preds_ptr, ground_truth_ptr
    preds_m, preds_ptr, n_rows, _, _ = \
        input_to_dev_array(predictions)

    ground_truth_m, ground_truth_ptr, _, _, ground_truth_dtype = \
        input_to_dev_array(ground_truth)

    acc = accuracy_score_py(handle_[0],
                            <int*> preds_ptr,
                            <int*> ground_truth_ptr,
                            <int> n_rows)

    return acc
