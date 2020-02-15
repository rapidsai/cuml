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
import cupy as cp

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


def homogeneity_score(labels_true, labels_pred, handle=None):
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
    labels_pred : int32 NumPy ndarray, int32 Numba device or int32 cudf Series
       The labels predicted by the model for the test dataset
    labels_true : int32 NumPy ndarray, int32 Numba device or int32 cudf Series
       The ground truth labels of the test dataset

    Returns
    -------
    float
      The homogeneity of the predicted labeling given the ground truth.
      Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.
    """
    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle*handle_ = <cumlHandle*> <size_t> handle.getHandle()

    cdef uintptr_t preds_ptr
    cdef uintptr_t ground_truth_ptr

    preds_m, preds_ptr, n_rows, _, _ = input_to_dev_array(
        labels_pred,
        convert_to_dtype=np.int32,
    )

    ground_truth_m, ground_truth_ptr, _, _, _ = input_to_dev_array(
        labels_true,
        convert_to_dtype=np.int32,
        check_rows=n_rows,
    )

    cp_ground_truth_m = cp.asarray(ground_truth_m)
    cp_preds_m = cp.asarray(preds_m)

    lower_class_range = min(cp.min(cp_ground_truth_m), cp.min(cp_preds_m))
    upper_class_range = max(cp.max(cp_ground_truth_m), cp.max(cp_preds_m))

    hom = homogeneityScore(handle_[0],
                           <int*> ground_truth_ptr,
                           <int*> preds_ptr,
                           <int> n_rows,
                           <int> lower_class_range,
                           <int> upper_class_range)

    return hom
