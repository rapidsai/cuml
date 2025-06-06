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

import cuml.internals

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle

from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double mutual_info_score(const handle_t &handle,
                             const int *y,
                             const int *y_hat,
                             const int n,
                             const int lower_class_range,
                             const int upper_class_range) except +


@cuml.internals.api_return_any()
def cython_mutual_info_score(labels_true, labels_pred, handle=None) -> float:
    """
    Computes the Mutual Information between two clusterings.

    The Mutual Information is a measure of the similarity between two labels of
    the same data.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score
    value in any way.

    This metric is furthermore symmetric: switching label_true with label_pred
    will return the same score value. This can be useful to measure the
    agreement of two independent label assignments strategies on the same
    dataset when the real ground truth is not known.

    The labels in labels_pred and labels_true are assumed to be drawn from a
    contiguous set (Ex: drawn from {2, 3, 4}, but not from {2, 4}). If your
    set of labels looks like {2, 4}, convert them to something like {0, 1}.

    Parameters
    ----------
    handle : cuml.Handle
    labels_pred : array-like (device or host) shape = (n_samples,)
        A clustering of the data (ints) into disjoint subsets.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    labels_true : array-like (device or host) shape = (n_samples,)
        A clustering of the data (ints) into disjoint subsets.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    Returns
    -------
    float
      Mutual information, a non-negative value
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    (y_true, y_pred, n_rows,
     lower_class_range, upper_class_range) = prepare_cluster_metric_inputs(
        labels_true,
        labels_pred
    )

    cdef uintptr_t ground_truth_ptr = y_true.ptr
    cdef uintptr_t preds_ptr = y_pred.ptr

    mi = mutual_info_score(handle_[0],
                           <int*> ground_truth_ptr,
                           <int*> preds_ptr,
                           <int> n_rows,
                           <int> lower_class_range,
                           <int> upper_class_range)

    return mi
