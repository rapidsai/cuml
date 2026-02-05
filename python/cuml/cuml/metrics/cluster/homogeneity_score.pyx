#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from cuml.internals import get_handle
from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    double homogeneity_score(const handle_t & handle, const int *y,
                             const int *y_hat, const int n,
                             const int lower_class_range,
                             const int upper_class_range) except +


def cython_homogeneity_score(labels_true, labels_pred) -> float:
    """
    Computes the homogeneity metric of a cluster labeling given a ground truth.

    A clustering result satisfies homogeneity if all of its clusters contain
    only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won’t change the score
    value in any way.

    This metric is not symmetric: switching label_true with label_pred will
    return the completeness_score which will be different in general.

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

    Returns
    -------
    float
      The homogeneity of the predicted labeling given the ground truth.
      Score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling.
    """
    handle = get_handle()
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    (y_true, y_pred,
     n_rows,
     lower_class_range, upper_class_range) = prepare_cluster_metric_inputs(
        labels_true,
        labels_pred
    )

    cdef uintptr_t ground_truth_ptr = y_true.ptr
    cdef uintptr_t preds_ptr = y_pred.ptr

    hom = homogeneity_score(handle_[0],
                            <int*> ground_truth_ptr,
                            <int*> preds_ptr,
                            <int> n_rows,
                            <int> lower_class_range,
                            <int> upper_class_range)

    return hom
