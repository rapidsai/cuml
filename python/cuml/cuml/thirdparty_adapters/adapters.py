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

import cudf
import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import coo_matrix as gpu_coo_matrix
from cupyx.scipy.sparse import csc_matrix as gpu_csc_matrix
from cupyx.scipy.sparse import csr_matrix as gpu_csr_matrix
from scipy import sparse as cpu_sparse
from scipy.sparse import csc_matrix as cpu_coo_matrix
from scipy.sparse import csc_matrix as cpu_csc_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import input_to_cupy_array, input_to_host_array

numeric_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.intp,
    np.uintp,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


def check_sparse(array, accept_sparse=False, accept_large_sparse=True):
    """Checks that the sparse array is valid

    Parameters
    ----------
    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

    Returns
    -------
    None or raise error
    """
    if accept_sparse is True:
        return

    err_msg = (
        "This algorithm does not support the sparse "
        + "input in the current configuration."
    )

    is_sparse = cpu_sparse.issparse(array) or gpu_sparse.issparse(array)
    if is_sparse:
        if accept_sparse is False:
            raise ValueError(err_msg)

        if not accept_large_sparse:
            if (
                array.indices.dtype != cp.int32
                or array.indptr.dtype != cp.int32
            ):
                raise ValueError(err_msg)

        if isinstance(accept_sparse, (tuple, list)):
            if array.format not in accept_sparse:
                raise ValueError(err_msg)
        elif array.format != accept_sparse:
            raise ValueError(err_msg)


def check_dtype(array, dtypes="numeric"):
    """Checks that the input dtype is part of acceptable dtypes

    Parameters
    ----------
    array : object
        Input object to check / convert.
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    Returns
    -------
    dtype or raise error
    """
    if dtypes is None:
        if not isinstance(array, cudf.DataFrame):
            return array.dtype
        else:
            return array.dtypes.tolist()[0]

    if dtypes == "numeric":
        dtypes = numeric_types

    if isinstance(dtypes, (list, tuple)):
        # fp16 is not supported, so remove from the list of dtypes if present
        dtypes = [d for d in dtypes if d != np.float16]

        if not isinstance(array, (pd.DataFrame, cudf.DataFrame)):
            if array.dtype not in dtypes:
                return dtypes[0]
        elif any([dt not in dtypes for dt in array.dtypes.tolist()]):
            return dtypes[0]

        if not isinstance(array, (pd.DataFrame, cudf.DataFrame)):
            return array.dtype
        else:
            return array.dtypes.tolist()[0]
    elif dtypes == np.float16:
        raise NotImplementedError("Float16 not supported by cuML")
    else:
        # Single dtype to convert to
        return dtypes


def check_finite(array, force_all_finite=True):
    """Checks that the input is finite if necessary

    Parameters
    ----------
    array : object
        Input object to check / convert.
    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    None or raise error
    """
    if force_all_finite is True:
        if not cp.all(cp.isfinite(array)):
            raise ValueError("Non-finite value encountered in array")
    elif force_all_finite == "allow-nan":
        if cp.any(cp.isinf(array)):
            raise ValueError("Non-finite value encountered in array")


def check_array(
    array,
    accept_sparse=False,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    warn_on_dtype=None,
    estimator=None,
):
    """Input validation on an array, list, sparse matrix or similar.
    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.
    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.
    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.
    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.
    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.
    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.
    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The
        possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
           ``force_all_finite`` accepts the string ``'allow-nan'``.
    ensure_2d : boolean (default=True)
        Whether to raise a value error if array is not 2D.
    allow_nd : boolean (default=False)
        Whether to allow array.ndim > 2.
    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.
    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.
    estimator : unused parameter

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """

    if dtype == "numeric":
        dtype = numeric_types

    correct_dtype = check_dtype(array, dtype)

    if (
        not isinstance(array, (pd.DataFrame, cudf.DataFrame))
        and copy
        and not order
        and hasattr(array, "flags")
    ):
        if array.flags["F_CONTIGUOUS"]:
            order = "F"
        elif array.flags["C_CONTIGUOUS"]:
            order = "C"

    if not order:
        order = "F"

    hasshape = hasattr(array, "shape")
    if ensure_2d and hasshape:
        if len(array.shape) != 2:
            raise ValueError("Not 2D")

    if not allow_nd and hasshape:
        if len(array.shape) > 2:
            raise ValueError("More than 2 dimensions detected")

    if ensure_min_samples > 0 and hasshape:
        if array.shape[0] < ensure_min_samples:
            raise ValueError("Not enough samples")

    if ensure_min_features > 0 and hasshape and len(array.shape) == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required."
                % (n_features, array.shape, ensure_min_features)
            )

    is_sparse = cpu_sparse.issparse(array) or gpu_sparse.issparse(array)
    if is_sparse:
        check_sparse(array, accept_sparse, accept_large_sparse)
        if array.format == "csr":
            if GlobalSettings().memory_type.is_device_accessible:
                new_array = gpu_csr_matrix(array, copy=copy)
            else:
                new_array = cpu_csr_matrix(array, copy=copy)
        elif array.format == "csc":
            if GlobalSettings().memory_type.is_device_accessible:
                new_array = gpu_csc_matrix(array, copy=copy)
            else:
                new_array = cpu_csc_matrix(array, copy=copy)
        elif array.format == "coo":
            if GlobalSettings().memory_type.is_device_accessible:
                new_array = gpu_coo_matrix(array, copy=copy)
            else:
                new_array = cpu_coo_matrix(array, copy=copy)
        else:
            raise ValueError("Sparse matrix format not supported")
        check_finite(new_array.data, force_all_finite)
        if correct_dtype != new_array.dtype:
            new_array = new_array.astype(correct_dtype)
        return new_array
    else:
        if GlobalSettings().memory_type.is_device_accessible:
            X, n_rows, n_cols, dtype = input_to_cupy_array(
                array, order=order, deepcopy=copy, fail_on_null=False
            )
        else:
            X, n_rows, n_cols, dtype = input_to_host_array(
                array, order=order, deepcopy=copy, fail_on_null=False
            )
        if correct_dtype != dtype:
            X = X.astype(correct_dtype)
        check_finite(X, force_all_finite)
        return X


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or cp.isnan(value_to_mask):
        return cp.isnan(X)
    else:
        return X == value_to_mask


def _masked_column_median(arr, masked_value):
    """Compute the median of each column in the 2D array arr, ignoring any
    instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    if arr.size == 0:
        return cp.full(arr.shape[1], cp.nan)
    if not cp.isnan(masked_value):
        arr_sorted = arr.copy()
        # If nan is not the missing value, any column with nans should
        # have a median of nan
        nan_cols = cp.any(cp.isnan(arr), axis=0)
        arr_sorted[mask] = cp.nan
        arr_sorted.sort(axis=0)
    else:
        nan_cols = cp.full(arr.shape[1], False)
        # nans are always sorted to end of array and the sort call
        # copies the data
        arr_sorted = cp.sort(arr, axis=0)

    count_missing_values = mask.sum(axis=0)
    # Ignore missing values in determining "halfway" index of sorted
    # array
    n_elems = arr.shape[0] - count_missing_values

    # If no elements remain after removing missing value, median for
    # that column is nan
    nan_cols = cp.logical_or(nan_cols, n_elems <= 0)

    col_index = cp.arange(arr_sorted.shape[1])
    median = (
        arr_sorted[cp.floor_divide(n_elems - 1, 2), col_index]
        + arr_sorted[cp.floor_divide(n_elems, 2), col_index]
    ) / 2

    median[nan_cols] = cp.nan
    return median


def _masked_column_mean(arr, masked_value):
    """Compute the mean of each column in the 2D array arr, ignoring any
    instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    count_missing_values = mask.sum(axis=0)
    n_elems = arr.shape[0] - count_missing_values
    mean = cp.nansum(arr, axis=0)
    if not cp.isnan(masked_value):
        mean -= count_missing_values * masked_value
    mean /= n_elems
    return mean


def _masked_column_mode(arr, masked_value):
    """Determine the most frequently appearing element in each column in the 2D
    array arr, ignoring any instances of masked_value"""
    mask = _get_mask(arr, masked_value)
    n_features = arr.shape[1]
    most_frequent = np.empty(n_features, dtype=arr.dtype)
    for i in range(n_features):
        feature_mask_idxs = cp.where(~mask[:, i])[0]
        values, counts = cp.unique(
            arr[feature_mask_idxs, i], return_counts=True
        )
        count_max = counts.max()
        if count_max > 0:
            value = values[counts == count_max].min()
        else:
            value = cp.nan
        most_frequent[i] = value
    return cp.array(most_frequent)
