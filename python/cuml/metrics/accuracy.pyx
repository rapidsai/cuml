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

import numpy as np

from libc.stdint cimport uintptr_t


import cuml.internals

from cuml.internals.input_utils import input_to_cuml_array
from pylibraft.common.handle cimport handle_t
from pylibraft.common.handle import Handle
cimport cuml.common.cuda

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":

    float accuracy_score_py(handle_t &handle,
                            int *predictions,
                            int *ref_predictions,
                            int n) except +


@cuml.internals.api_return_any()
def accuracy_score(ground_truth, predictions, handle=None, convert_dtype=True):
    """
    Calcuates the accuracy score of a classification model.

        Parameters
        ----------
        handle : cuml.Handle
        prediction : NumPy ndarray or Numba device
           The labels predicted by the model for the test dataset
        ground_truth : NumPy ndarray, Numba device
           The ground truth labels of the test dataset

        Returns
        -------
        float
          The accuracy of the model used for prediction
    """
    handle = Handle() \
        if handle is None else handle
    cdef handle_t* handle_ =\
        <handle_t*><size_t>handle.getHandle()

    cdef uintptr_t preds_ptr, ground_truth_ptr
    preds_m, n_rows, _, _ = \
        input_to_cuml_array(predictions,
                            convert_to_dtype=np.int32
                            if convert_dtype else None)

    preds_ptr = preds_m.ptr

    ground_truth_m, _, _, ground_truth_dtype=\
        input_to_cuml_array(ground_truth,
                            convert_to_dtype=np.int32
                            if convert_dtype else None)

    ground_truth_ptr = ground_truth_m.ptr

    acc = accuracy_score_py(handle_[0],
                            <int*> preds_ptr,
                            <int*> ground_truth_ptr,
                            <int> n_rows)

    return acc
