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

from cuml.common.handle cimport cumlHandle
from libc.stdint cimport uintptr_t

from cuml.metrics.cluster.mutual_info_utils import prepare_data
import cuml.common.handle


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

    The labels in labels_pred and labels_true are assumed to be drawn from a
    contiguous set (Ex: drawn from {2, 3, 4}, but not from {2, 4}). If your
    set of labels looks like {2, 4}, convert them to something like {0, 1}.

    Parameters
    ----------
    labels_pred : array-like (device or host) shape = (n_samples,)
        The labels predicted by the model for the test dataset.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    labels_true : array-like (device or host) shape = (n_samples,)
        The ground truth labels (ints) of the test dataset.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    float
      The homogeneity of the predicted labeling given the ground truth.
      Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.
    """
    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle *handle_ = <cumlHandle*> <size_t> handle.getHandle()

    cdef uintptr_t preds_ptr
    cdef uintptr_t ground_truth_ptr

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle *handle_ = <cumlHandle*> <size_t> handle.getHandle()

    (ground_truth_ptr, preds_ptr,
     n_rows,
     lower_class_range, upper_class_range) = prepare_data(labels_true,
                                                          labels_pred)

    hom = homogeneityScore(handle_[0],
                           <int*> ground_truth_ptr,
                           <int*> preds_ptr,
                           <int> n_rows,
                           <int> lower_class_range,
                           <int> upper_class_range)

    return hom
