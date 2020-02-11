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

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double homogeneityScore(const cumlHandle & handle, const int *y,
                            const int *y_hat, const int n,
                            const int lower_class_range,
                            const int upper_class_range) except +


def homogeneity_score(ground_truth, predictions, handle=None):
    """
    Computes the homogeneity metric of a cluster labeling given a ground truth.

    A clustering result satisfies homogeneity if all of its clusters contain
    only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score
    value in any way.

        Parameters
        ----------
        handle : cuml.Handle
        predictions : NumPy ndarray or Numba device
           The labels predicted by the model for the test dataset
        ground_truth : NumPy ndarray, Numba device
           The ground truth labels of the test dataset

        Returns
        -------
        float
          The homogeneity of the predicted labeling given the ground truth
    """
    handle = cuml.common.handle.Handle() \
        if handle is None else handle
    cdef cumlHandle*handle_ = \
        <cumlHandle*> <size_t> handle.getHandle()

    cdef uintptr_t preds_ptr, ground_truth_ptr
    preds_m, preds_ptr, n_rows, _, _ = \
        input_to_dev_array(predictions,
                           convert_to_dtype=
                           None
                           # np.int32
                           # if convert_dtype else None
                           )

    ground_truth_m, ground_truth_ptr, _, _, ground_truth_dtype = \
        input_to_dev_array(ground_truth,
                           convert_to_dtype=
                           None
                           # np.int32
                           # if convert_dtype else None
                           ,
                           check_rows=n_rows
                           )

    # TODO: Test when all labels are not in the ground_truth/preds, especially the min/max label
    lower_class_range = min(ground_truth_m.min(), preds_m.min())
    upper_class_range = max(ground_truth_m.max(), preds_m.max())

    hom = homogeneityScore(handle_[0],
                           <int*> ground_truth_ptr,
                           <int*> preds_ptr,
                           <int> n_rows,
                           lower_class_range,
                           upper_class_range)

    return hom
