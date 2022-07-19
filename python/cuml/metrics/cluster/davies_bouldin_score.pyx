#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from raft.common.handle cimport handle_t
from raft.common.handle import Handle
from cuml.metrics.distance_type cimport DistanceType
from cuml.prims.label.classlabels import make_monotonic, check_labels

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    double davies_bouldin_score(
            const handle_t &handle,
            double *y,
            int n_rows,
            int n_cols,
            int *labels,
            int n_labels,
            DistanceType metric) except +

def _davies_bouldin_measures(
        X, 
        labels, 
        metric='L2SqrtUnexpanded',
        handle=None):
    """Function wrapped by davies_bouldin_score to compute Davies Bouldin Scores.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        The feature vectors for all samples.
    labels : array-like, shape = (n_samples,)
        The assigned cluster labels for each sample.
    metric : string
        A string representation of the distance metric to use for evaluating
        the Davies Bouldin score. Available options are "cityblock", "cosine",
        "euclidean", "l1", "l2", "manhattan", and "sqeuclidean".
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

    data, n_rows, n_cols, dtype = input_to_cuml_array(
        X,
        order='C',
        check_dtype=np.float64,
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

    return davies_bouldin_score(handle_[0],
                                <double*> <uintptr_t> data.ptr,
                                <int> n_rows,
                                <int> n_cols,
                                <int*> <uintptr_t> mono_labels.ptr,
                                <int> n_labels,
                                <DistanceType> metric)


def cython_davies_bouldin_score(
        X,
        labels,
        metric='euclidean',
        handle=None):
    """Compute the Davies-Bouldin score.

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.
    The minimum score is zero, with lower values indicating better clustering.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape = (n_samples,)
        Assigned/Predicted labels for each sample.

    Returns
    -------
    score: double
        The resulting Davies-Bouldin score.

    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    
    return _davies_bouldin_measures(X, labels)