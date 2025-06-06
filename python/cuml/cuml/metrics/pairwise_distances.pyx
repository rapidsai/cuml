#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.common import CumlArray, input_to_cuml_array
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import sparse_scipy_to_cp

from cuml.metrics.distance_type cimport DistanceType

from cuml.thirdparty_adapters import _get_mask


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:
    void pairwise_distance(const handle_t &handle, const double *x,
                           const double *y, double *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor,
                           double metric_arg) except +
    void pairwise_distance(const handle_t &handle, const float *x,
                           const float *y, float *dist, int m, int n, int k,
                           DistanceType metric, bool isRowMajor,
                           float metric_arg) except +
    void pairwiseDistance_sparse(const handle_t &handle, float *x, float *y,
                                 float *dist, int x_nrows, int y_nrows,
                                 int n_cols, int x_nnz, int y_nnz,
                                 int* x_indptr, int* y_indptr,
                                 int* x_indices, int* y_indices,
                                 DistanceType metric,
                                 float metric_arg) except +
    void pairwiseDistance_sparse(const handle_t &handle, double *x, double *y,
                                 double *dist, int x_nrows, int y_nrows,
                                 int n_cols, int x_nnz, int y_nnz,
                                 int* x_indptr, int* y_indptr,
                                 int* x_indices, int* y_indices,
                                 DistanceType metric,
                                 float metric_arg) except +

# List of available distance metrics in `pairwise_distances`
PAIRWISE_DISTANCE_METRICS = {
    "cityblock": DistanceType.L1,
    "cosine": DistanceType.CosineExpanded,
    "euclidean": DistanceType.L2SqrtUnexpanded,
    "l1": DistanceType.L1,
    "l2": DistanceType.L2SqrtUnexpanded,
    "manhattan": DistanceType.L1,
    "sqeuclidean": DistanceType.L2Expanded,
    "canberra": DistanceType.Canberra,
    "chebyshev": DistanceType.Linf,
    "minkowski": DistanceType.LpUnexpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "correlation": DistanceType.CorrelationExpanded,
    "jensenshannon": DistanceType.JensenShannon,
    "hamming": DistanceType.HammingUnexpanded,
    "kldivergence": DistanceType.KLDivergence,
    "russellrao": DistanceType.RusselRaoExpanded,
    "nan_euclidean": DistanceType.L2Expanded
}

PAIRWISE_DISTANCE_SPARSE_METRICS = {
    "cityblock": DistanceType.L1,
    "cosine": DistanceType.CosineExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "l1": DistanceType.L1,
    "l2": DistanceType.L2SqrtExpanded,
    "manhattan": DistanceType.L1,
    "sqeuclidean": DistanceType.L2Expanded,
    "canberra": DistanceType.Canberra,
    "inner_product": DistanceType.InnerProduct,
    "minkowski": DistanceType.LpUnexpanded,
    "jaccard": DistanceType.JaccardExpanded,
    "hellinger": DistanceType.HellingerExpanded,
    "chebyshev": DistanceType.Linf,
    "dice": DistanceType.DiceExpanded
}


def _determine_metric(metric_str, is_sparse_=False):
    # Available options in scikit-learn and their pairs. See
    # sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS:
    # 'cityblock': L1
    # 'cosine': CosineExpanded
    # 'euclidean': L2SqrtUnexpanded
    # 'haversine': N/A
    # 'l2': L2SqrtUnexpanded
    # 'l1': L1
    # 'manhattan': L1
    # 'nan_euclidean': N/A
    # 'sqeuclidean': L2Unexpanded
    # Note: many are duplicates following this:
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/pairwise.py#L1321

    if metric_str == 'haversine':
        raise ValueError(" The metric: '{}', is not supported at this time."
                         .format(metric_str))

    if not is_sparse_ and (metric_str not in PAIRWISE_DISTANCE_METRICS):
        if metric_str in PAIRWISE_DISTANCE_SPARSE_METRICS:
            raise ValueError(" The metric: '{}', is only available on "
                             "sparse data.".format(metric_str))
        else:
            raise ValueError("Unknown metric: {}".format(metric_str))
    elif is_sparse_ and (metric_str not in PAIRWISE_DISTANCE_SPARSE_METRICS):
        raise ValueError("Unknown metric: {}".format(metric_str))

    if is_sparse_:
        return PAIRWISE_DISTANCE_SPARSE_METRICS[metric_str]
    else:
        return PAIRWISE_DISTANCE_METRICS[metric_str]


def nan_euclidean_distances(
    X, Y=None, *, squared=False, missing_values=cp.nan, convert_dtype=True
):
    """Calculate the euclidean distances in the presence of missing values.

    Compute the euclidean distance between each pair of samples in X and Y,
    where Y=X is assumed if Y=None. When calculating the distance between a
    pair of samples, this formulation ignores feature coordinates with a
    missing value in either sample and scales up the weight of the remaining
    coordinates:

        dist(x,y) = sqrt(weight * sq. distance from present coordinates)
        where,
        weight = Total # of coordinates / # of present coordinates

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
    is:

        .. math::
            \\sqrt{\\frac{4}{2}((3-1)^2 + (6-5)^2)}

    If all the coordinates are missing or if there are no common present
    coordinates then NaN is returned for that pair.

    Parameters
    ----------
    X : Dense matrix of shape (n_samples_X, n_features)
        Acceptable formats: cuDF DataFrame, Pandas DataFrame, NumPy ndarray,
        cuda array interface compliant array like CuPy.

    Y : Dense matrix of shape (n_samples_Y, n_features), default=None
        Acceptable formats: cuDF DataFrame, Pandas DataFrame, NumPy ndarray,
        cuda array interface compliant array like CuPy.

    squared : bool, default=False
        Return squared Euclidean distances.

    missing_values : np.nan or int, default=np.nan
        Representation of missing value.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.
    """

    if isinstance(X, cudf.DataFrame) or isinstance(X, pd.DataFrame):
        if (X.isnull().any()).any():
            X.fillna(0, inplace=True)

    if isinstance(Y, cudf.DataFrame) or isinstance(Y, pd.DataFrame):
        if (Y.isnull().any()).any():
            Y.fillna(0, inplace=True)

    X_m, _n_samples_x, _n_features_x, dtype_x = \
        input_to_cuml_array(X,
                            order="K",
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None),
                            check_dtype=[np.float32, np.float64])

    if Y is None:
        Y = X_m

    Y_m, _n_samples_y, _n_features_y, _dtype_y = \
        input_to_cuml_array(
            Y, order=X_m.order, convert_to_dtype=dtype_x,
            check_dtype=[dtype_x])

    X_m = cp.asarray(X_m)
    Y_m = cp.asarray(Y_m)

    # Get missing mask for X
    missing_X = _get_mask(X_m, missing_values)

    # Get missing mask for Y
    missing_Y = missing_X if Y is X else _get_mask(Y_m, missing_values)

    # set missing values to zero
    X_m[missing_X] = 0
    Y_m[missing_Y] = 0

    # Adjust distances for squared
    if X_m.shape == Y_m.shape:
        if (X_m == Y_m).all():
            distances = cp.asarray(pairwise_distances(
                X_m, metric="sqeuclidean"))
        else:
            distances = cp.asarray(pairwise_distances(
                X_m, Y_m, metric="sqeuclidean"))
    else:
        distances = cp.asarray(pairwise_distances(
            X_m, Y_m, metric="sqeuclidean"))

    # Adjust distances for missing values
    XX = X_m * X_m
    YY = Y_m * Y_m
    distances -= cp.dot(XX, missing_Y.T)
    distances -= cp.dot(missing_X, YY.T)

    cp.clip(distances, 0, None, out=distances)

    if X_m is Y_m:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        cp.fill_diagonal(distances, 0.0)

    present_X = 1 - missing_X
    present_Y = present_X if Y_m is X_m else ~missing_Y
    present_count = cp.dot(present_X, present_Y.T)
    distances[present_count == 0] = cp.nan

    # avoid divide by zero
    cp.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X_m.shape[1]

    if not squared:
        cp.sqrt(distances, out=distances)

    return distances


@cuml.internals.api_return_array(get_output_type=True)
def pairwise_distances(X, Y=None, metric="euclidean", handle=None,
                       convert_dtype=True, metric_arg=2, **kwds):
    """
    Compute the distance matrix from a vector array `X` and optional `Y`.

    This method takes either one or two vector arrays, and returns a distance
    matrix.

    If `Y` is given (default is `None`), then the returned matrix is the
    pairwise distance between the arrays from both `X` and `Y`.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', \
        'manhattan'].
        Sparse matrices are supported, see 'sparse_pairwise_distances'.
    - From scipy.spatial.distance: ['sqeuclidean']
        See the documentation for scipy.spatial.distance for details on this
        metric. Sparse matrices are supported.

    Parameters
    ----------
    X : Dense or sparse matrix (device or host) of shape
        (n_samples_x, n_features)
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy, or
        cupyx.scipy.sparse for sparse input

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

    >>> X = cp.array([[2.0, 3.0], [3.0, 5.0], [5.0, 8.0]])
    >>> Y = cp.array([[1.0, 0.0], [2.0, 1.0]])

    >>> # Euclidean Pairwise Distance, Single Input:
    >>> pairwise_distances(X, metric='euclidean')
    array([[0.        , 2.236..., 5.830...],
        [2.236..., 0.        , 3.605...],
        [5.830..., 3.605..., 0.        ]])

    >>> # Cosine Pairwise Distance, Multi-Input:
    >>> pairwise_distances(X, Y, metric='cosine')
    array([[0.445... , 0.131...],
        [0.485..., 0.156...],
        [0.470..., 0.146...]])

    >>> # Manhattan Pairwise Distance, Multi-Input:
    >>> pairwise_distances(X, Y, metric='manhattan')
    array([[ 4.,  2.],
        [ 7.,  5.],
        [12., 10.]])
    """

    if is_sparse(X):
        return sparse_pairwise_distances(X, Y, metric, handle,
                                         convert_dtype, **kwds)

    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    if metric in ['nan_euclidean']:
        return nan_euclidean_distances(X, Y, **kwds)

    if metric in ['russellrao'] and not np.all(X.data == 1.):
        warnings.warn("X was converted to boolean for metric {}"
                      .format(metric))
        X = np.where(X != 0., 1.0, 0.0)

    # Get the input arrays, preserve order and type where possible
    X_m, n_samples_x, n_features_x, dtype_x = \
        input_to_cuml_array(X,
                            order="K",
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None),
                            check_dtype=[np.float32, np.float64])

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

        if metric in ['russellrao'] and not np.all(Y.data == 1.):
            warnings.warn("Y was converted to boolean for metric {}"
                          .format(metric))
            Y = np.where(Y != 0., 1.0, 0.0)

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
                          <bool> is_row_major,
                          <float> metric_arg)
    elif (dtype_x == np.float64):
        pairwise_distance(handle_[0],
                          <double*> d_X_ptr,
                          <double*> d_Y_ptr,
                          <double*> d_dest_ptr,
                          <int> n_samples_x,
                          <int> n_samples_y,
                          <int> n_features_x,
                          <DistanceType> metric_val,
                          <bool> is_row_major,
                          <double> metric_arg)
    else:
        raise NotImplementedError("Unsupported dtype: {}".format(dtype_x))

    # Sync on the stream before exiting. pairwise_distance does not sync.
    handle.sync()

    del X_m
    del Y_m

    return dest_m


@cuml.internals.api_return_array(get_output_type=True)
def sparse_pairwise_distances(X, Y=None, metric="euclidean", handle=None,
                              convert_dtype=True, metric_arg=2, **kwds):
    """
    Compute the distance matrix from a vector array `X` and optional `Y`.

    This method takes either one or two sparse vector arrays, and returns a
    dense distance matrix.

    If `Y` is given (default is `None`), then the returned matrix is the
    pairwise distance between the arrays from both `X` and `Y`.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', \
        'manhattan'].
    - From scipy.spatial.distance: ['sqeuclidean', 'canberra', 'minkowski', \
        'jaccard', 'chebyshev', 'dice']
        See the documentation for scipy.spatial.distance for details on these
        metrics.
    - ['inner_product', 'hellinger']

    Parameters
    ----------
    X : array-like (device or host) of shape (n_samples_x, n_features)
        Acceptable formats: SciPy or Cupy sparse array

    Y : array-like (device or host) of shape (n_samples_y, n_features),\
        optional
        Acceptable formats: SciPy or Cupy sparse array

    metric : {"cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", \
        "sqeuclidean", "canberra", "lp", "inner_product", "minkowski", \
        "jaccard", "hellinger", "chebyshev", "linf", "dice"}
        The metric to use when calculating distance between instances in a
        feature array.

    convert_dtype : bool, optional (default = True)
        When set to True, the method will, when necessary, convert
        Y to be the same data type as X if they differ. This
        will increase memory used for the method.

    metric_arg : float, optional (default = 2)
        Additional metric-specific argument.
        For Minkowski it's the p-norm to apply.

    Returns
    -------
    D : array [n_samples_x, n_samples_x] or [n_samples_x, n_samples_y]
        A dense distance matrix D such that D_{i, j} is the distance between
        the ith and jth vectors of the given matrix `X`, if `Y` is None.
        If `Y` is not `None`, then D_{i, j} is the distance between the ith
        array from `X` and the jth array from `Y`.

    Examples
    --------

    .. code-block:: python

        >>> import cupyx
        >>> from cuml.metrics import sparse_pairwise_distances

        >>> X = cupyx.scipy.sparse.random(2, 3, density=0.5, random_state=9)
        >>> Y = cupyx.scipy.sparse.random(1, 3, density=0.5, random_state=9)
        >>> X.todense()
        array([[0.8098..., 0.537..., 0. ],
            [0.        , 0.856..., 0. ]])
        >>> Y.todense()
        array([[0.        , 0.        , 0.993...]])
        >>> # Cosine Pairwise Distance, Single Input:
        >>> sparse_pairwise_distances(X, metric='cosine')
        array([[0.      , 0.447...],
            [0.447..., 0.        ]])

        >>> # Squared euclidean Pairwise Distance, Multi-Input:
        >>> sparse_pairwise_distances(X, Y, metric='sqeuclidean')
        array([[1.931...],
            [1.720...]])

        >>> # Canberra Pairwise Distance, Multi-Input:
        >>> sparse_pairwise_distances(X, Y, metric='canberra')
        array([[3.],
            [2.]])
    """
    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
    if (not is_sparse(X)) or (Y is not None and not is_sparse(Y)):
        raise ValueError("Input matrices are not sparse.")

    dtype_x = X.data.dtype
    if dtype_x not in [cp.float32, cp.float64]:
        raise TypeError("Unsupported dtype: {}".format(dtype_x))

    if scipy.sparse.issparse(X):
        X = sparse_scipy_to_cp(X, dtype=None)

    if metric in ['jaccard', 'dice'] and not cp.all(X.data == 1.):
        warnings.warn("X was converted to boolean for metric {}"
                      .format(metric))
        X.data = (X.data != 0.).astype(dtype_x)

    X_m = SparseCumlArray(X)
    n_samples_x, n_features_x = X_m.shape
    if Y is None:
        Y_m = X_m
        dtype_y = dtype_x
    else:
        if scipy.sparse.issparse(Y):
            Y = sparse_scipy_to_cp(Y, dtype=dtype_x if convert_dtype else None)
        if convert_dtype:
            Y = Y.astype(dtype_x)
        elif dtype_x != Y.data.dtype:
            raise TypeError("Different data types unsupported when "
                            "convert_dtypes=False")

        if metric in ['jaccard', 'dice'] and not cp.all(Y.data == 1.):
            dtype_y = Y.data.dtype
            warnings.warn("Y was converted to boolean for metric {}"
                          .format(metric))
            Y.data = (Y.data != 0.).astype(dtype_y)
        Y_m = SparseCumlArray(Y)

    n_samples_y, n_features_y = Y_m.shape

    # Check feature sizes are equal
    if n_features_x != n_features_y:
        raise ValueError("Incompatible dimension for X and Y matrices: \
                         X.shape[1] == {} while Y.shape[1] == {}"
                         .format(n_features_x, n_features_y))

    # Get the metric string to a distance enum
    metric_val = _determine_metric(metric, is_sparse_=True)

    x_nrows, y_nrows = X_m.indptr.shape[0] - 1, Y_m.indptr.shape[0] - 1
    dest_m = CumlArray.zeros((x_nrows, y_nrows), dtype=dtype_x)
    cdef uintptr_t d_dest_ptr = dest_m.ptr

    cdef uintptr_t d_X_ptr = X_m.data.ptr
    cdef uintptr_t X_m_indptr = X_m.indptr.ptr
    cdef uintptr_t X_m_indices = X_m.indices.ptr

    cdef uintptr_t d_Y_ptr = Y_m.data.ptr
    cdef uintptr_t Y_m_indptr = Y_m.indptr.ptr
    cdef uintptr_t Y_m_indices = Y_m.indices.ptr

    if (dtype_x == np.float32):
        pairwiseDistance_sparse(handle_[0],
                                <float*> d_X_ptr,
                                <float*> d_Y_ptr,
                                <float*> d_dest_ptr,
                                <int> x_nrows,
                                <int> y_nrows,
                                <int> n_features_x,
                                <int> X_m.nnz,
                                <int> Y_m.nnz,
                                <int*> X_m_indptr,
                                <int*> Y_m_indptr,
                                <int*> X_m_indices,
                                <int*> Y_m_indices,
                                <DistanceType> metric_val,
                                <float> metric_arg)
    elif (dtype_x == np.float64):
        pairwiseDistance_sparse(handle_[0],
                                <double*> d_X_ptr,
                                <double*> d_Y_ptr,
                                <double*> d_dest_ptr,
                                <int> n_samples_x,
                                <int> n_samples_y,
                                <int> n_features_x,
                                <int> X_m.nnz,
                                <int> Y_m.nnz,
                                <int*> X_m_indptr,
                                <int*> Y_m_indptr,
                                <int*> X_m_indices,
                                <int*> Y_m_indices,
                                <DistanceType> metric_val,
                                <float> metric_arg)

    # Sync on the stream before exiting.
    handle.sync()

    del X_m
    del Y_m
    return dest_m
