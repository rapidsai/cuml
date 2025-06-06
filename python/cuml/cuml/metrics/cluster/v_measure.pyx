#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
    double v_measure(const handle_t & handle,
                     const int * y,
                     const int * y_hat,
                     const int n,
                     const int lower_class_range,
                     const int upper_class_range,
                     const double beta) except +


@cuml.internals.api_return_any()
def cython_v_measure(labels_true, labels_pred, beta=1.0, handle=None) -> float:
    """
    V-measure metric of a cluster labeling given a ground truth.

    The V-measure is the harmonic mean between homogeneity and completeness::

        v = (1 + beta) * homogeneity * completeness
             / (beta * homogeneity + completeness)

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

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
    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    v_measure_value : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
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

    v_measure_value = v_measure(handle_[0],
                                <int*> ground_truth_ptr,
                                <int*> preds_ptr,
                                <int> n_rows,
                                <int> lower_class_range,
                                <int> upper_class_range,
                                beta)

    return v_measure_value
