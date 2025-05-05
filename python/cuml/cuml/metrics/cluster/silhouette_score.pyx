#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np

from libc.stdint cimport uintptr_t

from cuml.common import input_to_cuml_array
from cuml.metrics.pairwise_distances import _determine_metric

from pylibraft.common.handle cimport handle_t

from pylibraft.common.handle import Handle

from cuml.metrics.distance_type cimport DistanceType

from cuml.prims.label.classlabels import check_labels, make_monotonic


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics::Batched" nogil:
    float silhouette_score(
        const handle_t &handle,
        float *y,
        int n_rows,
        int n_cols,
        int *labels,
        int n_labels,
        float *sil_scores,
        int chunk,
        DistanceType metric) except +

    double silhouette_score(
        const handle_t &handle,
        double *y,
        int n_rows,
        int n_cols,
        int *labels,
        int n_labels,
        double *sil_scores,
        int chunk,
        DistanceType metric) except +


def _silhouette_coeff(
        X, labels, metric='euclidean', sil_scores=None, chunksize=None,
        convert_dtype=True, handle=None):
    """Function wrapped by silhouette_score and silhouette_samples to compute
    silhouette coefficients.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    sil_scores : array_like, shape = (1, n_samples), dtype='float64'
        An optional array in which to store the silhouette score for each
        sample.
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    if chunksize is None:
        chunksize = 40000

    data, n_rows, n_cols, dtype = input_to_cuml_array(
        X,
        order='C',
        convert_to_dtype=(np.float32 if convert_dtype
                          else None),
        check_dtype=[np.float32, np.float64],
    )

    labels, _, _, _ = input_to_cuml_array(
        labels,
        order='C',
        convert_to_dtype=np.int32
    )

    n_labels = cp.unique(
        labels.to_output(output_type='cupy', output_dtype='int')
    ).shape[0]

    if not check_labels(labels, cp.arange(n_labels, dtype=np.int32)):
        mono_labels, _ = make_monotonic(labels, copy=True)
        mono_labels, _, _, _ = input_to_cuml_array(
            mono_labels,
            order='C',
            convert_to_dtype=np.int32
        )
    else:
        mono_labels = labels

    cdef uintptr_t scores_ptr
    if sil_scores is None:
        scores_ptr = <uintptr_t> NULL
    else:
        sil_scores = input_to_cuml_array(
            sil_scores,
            convert_to_dtype=(dtype if convert_dtype
                              else None),
            check_dtype=dtype)[0]

        scores_ptr = sil_scores.ptr

    metric = _determine_metric(metric)

    if dtype == np.float32:
        return silhouette_score(handle_[0],
                                <float*> <uintptr_t> data.ptr,
                                <int> n_rows,
                                <int> n_cols,
                                <int*> <uintptr_t> mono_labels.ptr,
                                <int> n_labels,
                                <float*> scores_ptr,
                                <int> chunksize,
                                <DistanceType> metric)
    elif dtype == np.float64:
        return silhouette_score(handle_[0],
                                <double*> <uintptr_t> data.ptr,
                                <int> n_rows,
                                <int> n_cols,
                                <int*> <uintptr_t> mono_labels.ptr,
                                <int> n_labels,
                                <double*> scores_ptr,
                                <int> chunksize,
                                <DistanceType> metric)


def cython_silhouette_score(
        X,
        labels,
        metric='euclidean',
        chunksize=None,
        convert_dtype=True,
        handle=None):
    """Calculate the mean silhouette coefficient for the provided data.

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    return _silhouette_coeff(
        X, labels, chunksize=chunksize, metric=metric,
        convert_dtype=convert_dtype, handle=handle
    )


def cython_silhouette_samples(
        X,
        labels,
        metric='euclidean',
        chunksize=None,
        convert_dtype=True,
        handle=None):
    """Calculate the silhouette coefficient for each sample in the provided data.

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    chunksize : integer (default = None)
        An integer, 1 <= chunksize <= n_samples to tile the pairwise distance
        matrix computations, so as to reduce the quadratic memory usage of
        having the entire pairwise distance matrix in GPU memory.
        If None, chunksize will automatically be set to 40000, which through
        experiments has proved to be a safe number for the computation
        to run on a GPU with 16 GB VRAM.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    sil_scores = cp.empty((X.shape[0],), dtype=X.dtype)

    _silhouette_coeff(
        X, labels, chunksize=chunksize, metric=metric, sil_scores=sil_scores,
        convert_dtype=convert_dtype, handle=handle
    )

    return sil_scores
