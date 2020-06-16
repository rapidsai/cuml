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

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from cuml.common.handle cimport cumlHandle
import cuml.common.handle
import cupy as cp
import numpy as np
from cuml.common.base import _input_to_type
from cuml.common import (get_cudf_column_ptr, get_dev_array_ptr,
                         input_to_cuml_array, CumlArray, logger, with_cupy_rmm)
from cuml.metrics.cluster.utils import prepare_cluster_metric_inputs

cdef extern from "metrics/trustworthiness_c.h" namespace "MLCommon::Distance":

    ctypedef int DistanceType
    ctypedef DistanceType euclidean "(MLCommon::Distance::DistanceType)5"

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
    void pairwiseDistance(const cumlHandle &handle, const double *x, const double *y, double *dist, int m,
                        int n, int k, int metric, bool isRowMajor) except +
    void pairwiseDistance(const cumlHandle &handle, const float *x, const float *y, float *dist, int m,
                        int n, int k, int metric, bool isRowMajor) except +

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

    This method takes either one or two vector arrays, and returns
    a distance matrix.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cosine', 'euclidean', 'l1', 'l2']. Sparse matrices are not supported.

    Parameters
    ----------
    X : array-like (device or host) shape = (n_samples_a, n_features)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    Y : array-like (device or host), optional shape = (n_samples_b, n_features)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    metric : string
        The metric to use when calculating distance between instances in a
        feature array.

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

    convert_dtype : bool, optional (default = True)
        When set to True, the fit method will, when necessary, convert
        Y to be the same data type as X if they differ. This
        will increase memory used for the method.

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

    # Determine the input type to convert to when returning
    output_type = _input_to_type(X)

    # Get the input arrays, preserve order and type where possible
    X_m, n_samples_x, n_features_x, dtype_x = \
        input_to_cuml_array(X, order="K", check_dtype=[np.float32, np.float64])
    
    # Get the order from the CumlArray
    input_order = X_m.order
    is_row_major = input_order == "C"

    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_Y_ptr
    cdef uintptr_t d_dest_ptr
    
    if (Y is not None):
        Y_m, n_samples_y, n_features_y, dtype_y = \
            input_to_cuml_array(Y, order="K", convert_to_dtype=(dtype_x if convert_dtype
                                              else None), check_dtype=[dtype_x])
    else:
        # Shallow copy X variables
        Y_m = X_m
        n_samples_y = n_samples_x
        n_features_y = n_features_x
        dtype_y = dtype_x

    # Check feature sizes are equal
    if (n_features_x != n_features_y):
        raise ValueError("Incompatible dimension for X and Y matrices: X.shape[1] == {} while Y.shape[1] == {}".format(n_features_x, n_features_y))

    if (X_m.order != Y_m.order):
        raise ValueError("Incompatible order for X and Y matrices: X.order == {} while Y.order == {}".format(X_m.order, Y_m.order))

    # Get the metric string to int
    metric_val = determine_metric(metric)

    # Create the output array
    dest_m = CumlArray.zeros((n_samples_x, n_samples_y), dtype=dtype_x, order=input_order)

    d_X_ptr = X_m.ptr
    d_Y_ptr = Y_m.ptr
    d_dest_ptr = dest_m.ptr

    # Now execute the functions
    if (dtype_x == np.float32):
        pairwiseDistance(handle_[0],
            <float*> d_X_ptr, 
            <float*> d_Y_ptr, 
            <float*> d_dest_ptr,
            <int> n_samples_x,
            <int> n_samples_y,
            <int> n_features_x,
            <int> metric_val,
            <bool> is_row_major)
    elif (dtype_x == np.float64):
        pairwiseDistance(handle_[0],
            <double*> d_X_ptr, 
            <double*> d_Y_ptr, 
            <double*> d_dest_ptr,
            <int> n_samples_x,
            <int> n_samples_y,
            <int> n_features_x,
            <int> metric_val,
            <bool> is_row_major)
    else:
        raise NotImplementedError("Unsupported dtype: {}".format(dtype_x))

    del X_m
    del Y_m

    return dest_m.to_output(output_type)