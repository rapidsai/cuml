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

from libc.stdint cimport uintptr_t
from cuml.common.handle cimport cumlHandle
import cuml.common.handle
import cupy as cp
import numpy as np
from cuml.common import (get_cudf_column_ptr, get_dev_array_ptr,
                         input_to_cuml_array, CumlArray, logger, with_cupy_rmm)
from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs

cdef extern from "metrics/trustworthiness_c.h" namespace "MLCommon::Distance":

    ctypedef int DistanceType
    ctypedef DistanceType euclidean "(MLCommon::Distance::DistanceType)5"

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    void pairwiseDistance(const cumlHandle &handle, const double *x, const double *y, double *dist, int m,
                        int n, int k, int metric) except +
    void pairwiseDistance(const cumlHandle &handle, const float *x, const float *y, float *dist, int m,
                        int n, int k, int metric) except +

def determine_metric(metric_str):

    # Available options in scikit-learn and their pairs. See sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS:
    # 'cityblock': N/A
    # 'cosine': EucExpandedCosine
    # 'euclidean': EucUnexpandedL2Sqrt
    # 'haversine': N/A
    # 'l2': EucUnexpandedL2
    # 'l1': EucUnexpandedL1
    # 'manhattan': N/A
    # 'nan_euclidean': N/A

    # TODO: Pull int values from actual enum
    if metric_str == 'cosine':
        return 2
    elif metric_str == 'euclidean':
        return 5
    elif metric_str == 'l2':
        return 1
    elif metric_str == '0':
        return 0
    elif metric_str == '4':
        return 4
    elif metric_str == '2':
        return 2
    elif metric_str == 'l1':
        return 3
    else:
        raise ValueError("Unknown metric: {}".format(metric_str))


@with_cupy_rmm
def pairwise_distances(X, Y=None, metric="euclidean", force_all_finite=True, handle=None, convert_dtype=True, **kwds):
    """ 
    Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix
      inputs.
      ['nan_euclidean'] but it does not yet support sparse matrices.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.22
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `np.nan`

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See also
    --------
    pairwise_distances_chunked : performs the same calculation as this
        function, but returns a generator of chunks of the distance matrix, in
        order to limit memory usage.
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle *handle_ = <cumlHandle*> <size_t> handle.getHandle()

    # Get the input arrays
    X_m, n_samples_x, n_features_x, dtype_x = \
        input_to_cuml_array(X, order='C', check_dtype=[np.float32, np.float64])
    
    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_Y_ptr
    cdef uintptr_t d_dest_ptr
    
    with logger.set_level(logger.LEVEL_DEBUG):

        logger.debug("Input Vals: {}, {}, {}".format(n_features_x, n_features_x, dtype_x))

        if (Y is not None):
            Y_m, n_samples_y, n_features_y, dtype_y = \
                input_to_cuml_array(Y, order='C', convert_to_dtype=dtype_x)
        else:
            # Shallow copy X variables
            Y_m = X_m
            n_samples_y = n_samples_x
            n_features_y = n_features_x
            dtype_y = dtype_x

        # TODO: Assert the n_features_x == n_features_y

        metric_val = determine_metric(metric)

        # Create the output array
        dest_m = CumlArray.zeros((n_samples_x, n_samples_y), dtype=dtype_x, order='C')

        d_X_ptr = X_m.ptr
        d_Y_ptr = Y_m.ptr
        d_dest_ptr = dest_m.ptr

        if (dtype_x == np.float32):
            pairwiseDistance(handle_[0],
                <float*> d_X_ptr, 
                <float*> d_Y_ptr, 
                <float*> d_dest_ptr,
                <int> n_samples_x,
                <int> n_samples_y,
                <int> n_features_x,
                <int> metric_val)
        elif (dtype_x == np.float64):
            pairwiseDistance(handle_[0],
                <double*> d_X_ptr, 
                <double*> d_Y_ptr, 
                <double*> d_dest_ptr,
                <int> n_samples_x,
                <int> n_samples_y,
                <int> n_features_x,
                <int> metric_val)
        else:
            # TODO: raise error. Should never get here
            pass

        del X_m
        del Y_m

        return dest_m.to_output("numpy")
