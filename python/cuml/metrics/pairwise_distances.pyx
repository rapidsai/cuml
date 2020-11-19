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

# distutils: language = c++

import warnings

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from cuml.raft.common.handle cimport handle_t
from cuml.raft.common.handle import Handle
import cupy as cp
import numpy as np
import cuml.internals
from cuml.common.base import _determine_stateless_output_type
from cuml.common import (input_to_cuml_array, CumlArray, logger)
from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs

cdef extern from "raft/linalg/distance_type.h" namespace "raft::distance":

    cdef enum DistanceType:
        EucExpandedL2 "raft::distance::DistanceType::EucExpandedL2"
        EucExpandedL2Sqrt "raft::distance::DistanceType::EucExpandedL2Sqrt"
        EucExpandedCosine "raft::distance::DistanceType::EucExpandedCosine"
        EucUnexpandedL1 "raft::distance::DistanceType::EucUnexpandedL1"
        EucUnexpandedL2 "raft::distance::DistanceType::EucUnexpandedL2"
        EucUnexpandedL2Sqrt "raft::distance::DistanceType::EucUnexpandedL2Sqrt"

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    void pairwise_distance(const handle_t &handle, const double *x,
                           const double *y, double *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor) except +
    void pairwise_distance(const handle_t &handle, const float *x,
                           const float *y, float *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor) except +


"""
List of available distance metrics in `pairwise_distances`
"""
PAIRWISE_DISTANCE_METRICS = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "sqeuclidean"
]


def _determine_metric(metric_str):

    # Available options in scikit-learn and their pairs. See
    # sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS:
    # 'cityblock': EucUnexpandedL1
    # 'cosine': EucExpandedCosine
    # 'euclidean': EucUnexpandedL2Sqrt
    # 'haversine': N/A
    # 'l2': EucUnexpandedL2Sqrt
    # 'l1': EucUnexpandedL1
    # 'manhattan': EucUnexpandedL1
    # 'nan_euclidean': N/A
    # 'sqeuclidean': EucUnexpandedL2
    # Note: many are duplicates following this:
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/pairwise.py#L1321

    if metric_str == 'cityblock':
        return DistanceType.EucUnexpandedL1
    elif metric_str == 'cosine':
        return DistanceType.EucExpandedCosine
    elif metric_str == 'euclidean':
        return DistanceType.EucUnexpandedL2Sqrt
    elif metric_str == 'haversine':
        raise ValueError(" The metric: '{}', is not supported at this time."
                         .format(metric_str))
    elif metric_str == 'l2':
        return DistanceType.EucUnexpandedL2Sqrt
    elif metric_str == 'l1':
        return DistanceType.EucUnexpandedL1
    elif metric_str == 'manhattan':
        return DistanceType.EucUnexpandedL1
    elif metric_str == 'nan_euclidean':
        raise ValueError(" The metric: '{}', is not supported at this time."
                         .format(metric_str))
    elif metric_str == 'sqeuclidean':
        return DistanceType.EucUnexpandedL2
    else:
        raise ValueError("Unknown metric: {}".format(metric_str))


@cuml.internals.api_return_array(get_output_type=True)
def pairwise_distances(X, Y=None, metric="euclidean", handle=None,
                       convert_dtype=True, output_type=None, **kwds):
    """
    Compute the distance matrix from a vector array `X` and optional `Y`.

    This method takes either one or two vector arrays, and returns a distance
    matrix.

    If `Y` is given (default is `None`), then the returned matrix is the
    pairwise distance between the arrays from both `X` and `Y`.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', \
        'manhattan'].
        Sparse matrices are not supported.
    - From scipy.spatial.distance: ['sqeuclidean']
        See the documentation for scipy.spatial.distance for details on this
        metric. Sparse matrices are not supported.

    Parameters
    ----------
    X : array-like (device or host) of shape (n_samples_x, n_features)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    Y : array-like (device or host) of shape (n_samples_y, n_features),\
        optional
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    metric : {"cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", \
        "sqeuclidean"}
        The metric to use when calculating distance between instances in a
        feature array.

    convert_dtype : bool, optional (default = True)
        When set to True, the method will, when necessary, convert
        Y to be the same data type as X if they differ. This
        will increase memory used for the method.

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_output_type`.
        See :ref:`output-data-type-configuration` for more info.

        .. deprecated:: 0.17
           `output_type` is deprecated in 0.17 and will be removed in 0.18.
           Please use the module level output type control,
           `cuml.global_output_type`.
           See :ref:`output-data-type-configuration` for more info.

    Returns
    -------
    D : array [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix `X`, if `Y` is None.
        If `Y` is not `None`, then D_{i, j} is the distance between the ith
        array from `X` and the jth array from `Y`.

    Examples
    --------
        >>> import cupy as cp
        >>> from cuml.metrics import pairwise_distances
        >>>
        >>> X = cp.array([[2.0, 3.0], [3.0, 5.0], [5.0, 8.0]])
        >>> Y = cp.array([[1.0, 0.0], [2.0, 1.0]])
        >>>
        >>> # Euclidean Pairwise Distance, Single Input:
        >>> pairwise_distances(X, metric='euclidean')
        array([[0.        , 2.23606798, 5.83095189],
            [2.23606798, 0.        , 3.60555128],
            [5.83095189, 3.60555128, 0.        ]])
        >>>
        >>> # Cosine Pairwise Distance, Multi-Input:
        >>> pairwise_distances(X, Y, metric='cosine')
        array([[0.4452998 , 0.13175686],
            [0.48550424, 0.15633851],
            [0.47000106, 0.14671817]])
        >>>
        >>> # Manhattan Pairwise Distance, Multi-Input:
        >>> pairwise_distances(X, Y, metric='manhattan')
        array([[ 4.,  2.],
            [ 7.,  5.],
            [12., 10.]])
    """

    # Check for deprecated `output_type` and warn. Set manually if specified
    if (output_type is not None):
        warnings.warn("Using the `output_type` argument is deprecated and "
                      "will be removed in 0.18. Please specify the output "
                      "type using `cuml.using_output_type()` instead",
                      DeprecationWarning)

        cuml.internals.set_api_output_type(output_type)

    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    # Get the input arrays, preserve order and type where possible
    X_m, n_samples_x, n_features_x, dtype_x = \
        input_to_cuml_array(X, order="K", check_dtype=[np.float32, np.float64])

    # Get the order from the CumlArray
    input_order = X_m.order

    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_Y_ptr
    cdef uintptr_t d_dest_ptr

    if (Y is not None):

        # Check for the odd case where one dimension of X is 1. In this case,
        # CumlArray always returns order=="C" so instead get the order from Y
        if (n_samples_x == 1 or n_features_x == 1):
            input_order = "K"

        Y_m, n_samples_y, n_features_y, dtype_y = \
            input_to_cuml_array(Y, order=input_order,
                                convert_to_dtype=(dtype_x if convert_dtype
                                                  else None),
                                check_dtype=[dtype_x])

        # Get the order from Y if necessary (It's possible to set order="F" in
        # input_to_cuml_array and have Y_m.order=="C")
        if (input_order == "K"):
            input_order = Y_m.order
    else:
        # Shallow copy X variables
        Y_m = X_m
        n_samples_y = n_samples_x
        n_features_y = n_features_x
        dtype_y = dtype_x

    is_row_major = input_order == "C"

    # Check feature sizes are equal
    if (n_features_x != n_features_y):
        raise ValueError("Incompatible dimension for X and Y matrices: \
                         X.shape[1] == {} while Y.shape[1] == {}"
                         .format(n_features_x, n_features_y))

    # Get the metric string to int
    metric_val = _determine_metric(metric)

    # Create the output array
    dest_m = CumlArray.zeros((n_samples_x, n_samples_y), dtype=dtype_x,
                             order=input_order)

    d_X_ptr = X_m.ptr
    d_Y_ptr = Y_m.ptr
    d_dest_ptr = dest_m.ptr

    # Now execute the functions
    if (dtype_x == np.float32):
        pairwise_distance(handle_[0],
                          <float*> d_X_ptr,
                          <float*> d_Y_ptr,
                          <float*> d_dest_ptr,
                          <int> n_samples_x,
                          <int> n_samples_y,
                          <int> n_features_x,
                          <DistanceType> metric_val,
                          <bool> is_row_major)
    elif (dtype_x == np.float64):
        pairwise_distance(handle_[0],
                          <double*> d_X_ptr,
                          <double*> d_Y_ptr,
                          <double*> d_dest_ptr,
                          <int> n_samples_x,
                          <int> n_samples_y,
                          <int> n_features_x,
                          <DistanceType> metric_val,
                          <bool> is_row_major)
    else:
        raise NotImplementedError("Unsupported dtype: {}".format(dtype_x))

    # Sync on the stream before exiting. pairwise_distance does not sync.
    handle.sync()

    del X_m
    del Y_m

    return dest_m
