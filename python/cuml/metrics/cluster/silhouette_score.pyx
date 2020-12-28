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

import cupy as cp
import numpy as np

from libc.stdint cimport uintptr_t

from cuml.common import input_to_cuml_array
from cuml.metrics.pairwise_distances import _determine_metric
from cuml.raft.common.handle cimport handle_t
from cuml.raft.common.handle import Handle
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double silhouette_score(
        const handle_t &handle,
        double *y,
        int n_rows,
        int n_cols,
        int *labels,
        int n_labels,
        double *sil_scores,
        DistanceType metric) except +


def _silhouette_coeff(
        X, labels, metric='euclidean', sil_scores=None, handle=None):
    """Function wrapped by silhouette_score and silhouette_samples to compute
    silhouette coefficients

    Warning
    -------
    The underlying silhouette_score implementation's memory usage is quadratic
    in the number of samples, so this call will fail on anything more than a
    modest-size input (relative to available GPU memory). This issue is being
    tracked at https://github.com/rapidsai/cuml/issues/3255 and will be fixed
    in an upcoming release.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette schore. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    sil_scores : array_like, shape = (1, n_samples), dtype='float64'
        An optional array in which to store the silhouette score for each
        sample.
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

    data, n_rows, n_cols, _ = input_to_cuml_array(
        X,
        order='C',
        convert_to_dtype=np.float64
    )

    labels, _, _, _ = input_to_cuml_array(
        labels,
        order='C',
        convert_to_dtype=np.int32
    )

    n_labels = cp.unique(
        labels.to_output(output_type='cupy', output_dtype='int')
    ).shape[0]

    cdef uintptr_t scores_ptr
    if sil_scores is None:
        scores_ptr = <uintptr_t> NULL
    else:
        sil_scores = input_to_cuml_array(
            sil_scores,
            check_dtype=np.float64)[0]

        scores_ptr = sil_scores.ptr

    metric = _determine_metric(metric)

    return silhouette_score(handle_[0],
                            <double*> <uintptr_t> data.ptr,
                            n_rows,
                            n_cols,
                            <int*> <uintptr_t> labels.ptr,
                            n_labels,
                            <double*> scores_ptr,
                            metric)


def cython_silhouette_score(
        X,
        labels,
        metric='euclidean',
        handle=None):
    """Calculate the mean silhouette coefficient for the provided data

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Warning
    -------
    The underlying silhouette_score implementation's memory usage is quadratic
    in the number of samples, so this call will fail on anything more than a
    modest-size input (relative to available GPU memory). This issue is being
    tracked at https://github.com/rapidsai/cuml/issues/3255 and will be fixed
    in an upcoming release.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette schore. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    return _silhouette_coeff(
        X, labels, metric=metric, handle=handle
    )


def cython_silhouette_samples(
        X,
        labels,
        metric='euclidean',
        handle=None):
    """Calculate the silhouette coefficient for each sample in the provided data

    Given a set of cluster labels for every sample in the provided data,
    compute the mean intra-cluster distance (a) and the mean nearest-cluster
    distance (b) for each sample. The silhouette coefficient for a sample is
    then (b - a) / max(a, b).

    Warning
    -------
    The underlying silhouette_score implementation's memory usage is quadratic
    in the number of samples, so this call will fail on anything more than a
    modest-size input (relative to available GPU memory). This issue is being
    tracked at https://github.com/rapidsai/cuml/issues/3255 and will be fixed
    in an upcoming release.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the silhouette schore. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    """

    sil_scores = cp.empty((X.shape[0],), dtype='float64')

    _silhouette_coeff(
        X, labels, metric=metric, sil_scores=sil_scores, handle=handle
    )

    return sil_scores
