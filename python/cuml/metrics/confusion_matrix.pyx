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

import numpy as np
import cupy as cp

from libc.stdint cimport uintptr_t

import cuml.common.handle
from cuml.utils import input_to_cuml_array
from cuml.common.handle cimport cumlHandle
from cuml.utils.memory_utils import with_cupy_rmm
from cuml.common import CumlArray
from cuml.metrics.utils import sorted_unique_labels


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    void contingencyMatrix(const cumlHandle &handle,
                           const int *groundTruth, const int *predictedLabel,
                           const int nSamples, int *outMat) except +


@with_cupy_rmm
def confusion_matrix(y_true, y_pred,
                     labels=None,
                     sample_weight=None,
                     normalize=None,
                     use_cuda_coo=False,
                     handle=None):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
        Estimated target values.
    labels : array-like (device or host) shape = (n_classes,), optional
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None is given, those that appear at least
        once in y_true or y_pred are used in sorted order.
    sample_weight : array-like (device or host) shape = (n_samples,), optional
        Sample weights.
    normalize : string in [‘true’, ‘pred’, ‘all’]
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    use_cuda_coo : bool, optional
        Whether or not to use our cuda version of the coo function.
        If false, will use cupy's version of coo.
        Used for benchmarks.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this function. Most importantly, this specifies the
        CUDA stream that will be used for this function's computations, so
        users can run different computations concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.

    Returns
    -------
    C : array-like (device or host) shape = (n_classes, n_classes)
        Confusion matrix.
    """
    handle = cuml.common.handle.Handle() \
        if handle is None else handle
    cdef cumlHandle* handle_ =\
        <cumlHandle*><size_t>handle.getHandle()

    y_true, n_rows, n_cols, dtype = \
        input_to_cuml_array(y_true, check_dtype=[cp.int32, cp.int64])

    y_pred, _, _, _ = \
        input_to_cuml_array(y_pred, check_dtype=dtype,
                            check_rows=n_rows, check_cols=n_cols)

    if labels is None:
        labels = sorted_unique_labels(y_true, y_pred)
    else:
        labels, n_labels, _, _ = \
            input_to_cuml_array(labels, check_dtype=dtype, check_cols=1)
        if cp.all([l not in y_true for l in labels]):
            raise ValueError("At least one label specified must be in y_true")

    if sample_weight is None:
        sample_weight = cp.ones(n_rows, dtype=dtype)
    else:
        if use_cuda_coo:
            raise NotImplementedError("Sample weights not implemented with "
                                      "cuda coo.")
        sample_weight, _, _, _ = \
            input_to_cuml_array(sample_weight, check_dtype=dtype,
                                check_rows=n_rows, check_cols=n_cols)

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    label_to_ind = {y: x for x, y in enumerate(labels)}

    y_pred = cp.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = cp.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = cp.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if dtype.kind in {'i', 'u', 'b'}:
        dtype = cp.int64
    else:
        dtype = cp.float64

    if use_cuda_coo:
        cm = CumlArray.zeros(shape=(n_labels, n_labels), dtype=dtype,
                             order='C')
        cdef uintptr_t cm_ptr = cm.ptr
        cdef uintptr_t y_pred_ptr = y_pred.ptr
        cdef uintptr_t y_true_ptr = y_true.ptr

        contingencyMatrix(handle_[0],
                          <int*> y_true_ptr,
                          <int*> y_pred_ptr,
                          <int> n_rows,
                          <int*> cm_ptr)
        # TODO: Implement weighting
    else:
        cm = cp.sparse.coo_matrix((sample_weight, (y_true, y_pred)),
                                  shape=(n_labels, n_labels), dtype=dtype,
                                  ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cp.divide(cm, cm.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            cm = cp.divide(cm, cm.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            cm = cp.divide(cm, cm.sum())
        cm = cp.nan_to_num(cm)

    return cm
