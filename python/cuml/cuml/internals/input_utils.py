#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from collections import namedtuple

import cudf
import cupy as cp
import cupyx
import numba.cuda as numba_cuda
import numpy as np
import pandas as pd
from packaging.version import Version
from pandas.api.types import is_extension_array_dtype, is_string_dtype

import cuml.internals.nvtx as nvtx
from cuml.internals.array import CumlArray
from cuml.internals.mem_type import MemoryType

PANDAS_VERSION = Version(pd.__version__)

cuml_array = namedtuple("cuml_array", "array n_rows n_cols dtype")


def determine_array_dtype(X):
    if X is None:
        return None

    if isinstance(X, (cudf.DataFrame, pd.DataFrame)):
        # Assume single-label target
        dtype = X[X.columns[0]].dtype
    else:
        try:
            dtype = X.dtype
        except AttributeError:
            dtype = None

    if dtype is not None and (
        is_string_dtype(dtype) or is_extension_array_dtype(dtype)
    ):
        return np.dtype("object")

    return dtype


@nvtx.annotate(
    message="common.input_utils.input_to_cuml_array",
    category="utils",
    domain="cuml_python",
)
def input_to_cuml_array(
    X,
    order="F",
    deepcopy=False,
    check_dtype=False,
    convert_to_dtype=False,
    check_mem_type=False,
    convert_to_mem_type="device",
    safe_dtype_conversion=True,
    check_cols=False,
    check_rows=False,
    fail_on_order=False,
    force_contiguous=True,
):
    """
    Convert input X to CumlArray.

    Acceptable input formats:

    * cuDF Dataframe - returns a deep copy always.
    * cuDF Series - returns by reference or a deep copy depending on
        `deepcopy`.
    * Numpy array - returns a copy in device always
    * cuda array interface compliant array (like Cupy) - returns a
        reference unless `deepcopy`=True.
    * numba device array - returns a reference unless deepcopy=True

    Parameters
    ----------

    X : cuDF.DataFrame, cuDF.Series, NumPy array, Pandas DataFrame, Pandas
        Series or any cuda_array_interface (CAI) compliant array like CuPy,
        Numba or pytorch.

    order: 'F', 'C' or 'K' (default: 'F')
        Whether to return a F-major ('F'),  C-major ('C') array or Keep ('K')
        the order of X. Used to check the order of the input. If
        fail_on_order=True, the method will raise ValueError,
        otherwise it will convert X to be of order `order` if needed.

    deepcopy: boolean (default: False)
        Set to True to always return a deep copy of X.

    check_dtype: np.dtype (default: False)
        Set to a np.dtype to throw an error if X is not of dtype `check_dtype`.

    convert_to_dtype: np.dtype (default: False)
        Set to a dtype if you want X to be converted to that dtype if it is
        not that dtype already.

    safe_convert_to_dtype: bool (default: True)
        Set to True to check whether a typecasting performed when
        convert_to_dtype is True will cause information loss. This has a
        performance implication that might be significant for very fast
        methods like FIL and linear models inference.

    check_cols: int (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    check_rows: boolean (default: False)
        Set to an int `i` to check that input X has `i` rows. Set to False
        (default) to not check at all.

    fail_on_order: boolean (default: False)
        Set to True if you want the method to raise a ValueError if X is not
        of order `order`.

    force_contiguous: boolean (default: True)
        Set to True to force CumlArray produced to be contiguous. If `X` is
        non contiguous then a contiguous copy will be done.
        If False, and `X` doesn't need to be converted and is not contiguous,
        the underlying memory underneath the CumlArray will be non contiguous.
        Only affects CAI inputs. Only affects CuPy and Numba device array
        views, all other input methods produce contiguous CumlArrays.

    Returns
    -------
    `cuml_array`: namedtuple('cuml_array', 'array n_rows n_cols dtype')

        A new CumlArray and associated data.

    """
    arr = CumlArray.from_input(
        X,
        order=order,
        deepcopy=deepcopy,
        check_dtype=check_dtype,
        convert_to_dtype=convert_to_dtype,
        check_mem_type=check_mem_type,
        convert_to_mem_type=convert_to_mem_type,
        safe_dtype_conversion=safe_dtype_conversion,
        check_cols=check_cols,
        check_rows=check_rows,
        fail_on_order=fail_on_order,
        force_contiguous=force_contiguous,
    )
    try:
        shape = arr.__cuda_array_interface__["shape"]
    except AttributeError:
        shape = arr.__array_interface__["shape"]

    n_rows = shape[0]

    if len(shape) > 1:
        n_cols = shape[1]
    else:
        n_cols = 1

    return cuml_array(array=arr, n_rows=n_rows, n_cols=n_cols, dtype=arr.dtype)


@nvtx.annotate(
    message="common.input_utils.input_to_cupy_array",
    category="utils",
    domain="cuml_python",
)
def input_to_cupy_array(
    X,
    order="F",
    deepcopy=False,
    check_dtype=False,
    convert_to_dtype=False,
    check_cols=False,
    check_rows=False,
    fail_on_order=False,
    force_contiguous=True,
    fail_on_null=True,
) -> cuml_array:
    """
    Identical to input_to_cuml_array but it returns a cupy array instead of
    CumlArray
    """
    if not fail_on_null:
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            try:
                X = X.values
            except ValueError:
                X = X.astype("float64", copy=False)
                X.fillna(cp.nan, inplace=True)
                X = X.values

    out_data = input_to_cuml_array(
        X,
        order=order,
        deepcopy=deepcopy,
        check_dtype=check_dtype,
        convert_to_dtype=convert_to_dtype,
        check_cols=check_cols,
        check_rows=check_rows,
        fail_on_order=fail_on_order,
        force_contiguous=force_contiguous,
        convert_to_mem_type=MemoryType.device,
    )

    return out_data._replace(array=out_data.array.to_output("cupy"))


@nvtx.annotate(
    message="common.input_utils.input_to_host_array",
    category="utils",
    domain="cuml_python",
)
def input_to_host_array(
    X,
    order="F",
    deepcopy=False,
    check_dtype=False,
    convert_to_dtype=False,
    check_cols=False,
    check_rows=False,
    fail_on_order=False,
    force_contiguous=True,
    fail_on_null=True,
) -> cuml_array:
    """
    Identical to input_to_cuml_array but it returns a host (NumPy array instead
    of CumlArray
    """
    if not fail_on_null and isinstance(X, (cudf.DataFrame, cudf.Series)):
        try:
            X = X.values
        except ValueError:
            X = X.astype("float64", copy=False)
            X.fillna(cp.nan, inplace=True)
            X = X.values

    out_data = input_to_cuml_array(
        X,
        order=order,
        deepcopy=deepcopy,
        check_dtype=check_dtype,
        convert_to_dtype=convert_to_dtype,
        check_cols=check_cols,
        check_rows=check_rows,
        fail_on_order=fail_on_order,
        force_contiguous=force_contiguous,
        convert_to_mem_type=MemoryType.host,
    )

    return out_data._replace(array=out_data.array.to_output("numpy"))


def convert_dtype(X, to_dtype=np.float32, legacy=True, safe_dtype=True):
    """
    Convert X to be of dtype `dtype`, raising a TypeError
    if the conversion would lose information.
    """

    if hasattr(X, "__dask_graph__") and hasattr(X, "compute"):
        # TODO: Warn, but not when using dask_sql
        X = X.compute()

    if safe_dtype:
        cur_dtype = determine_array_dtype(X)
        if not np.can_cast(cur_dtype, to_dtype):
            try:
                target_dtype_range = cp.iinfo(to_dtype)
            except ValueError:
                target_dtype_range = cp.finfo(to_dtype)
            out_of_range = (
                (X < target_dtype_range.min) | (X > target_dtype_range.max)
            ).any()
            try:
                out_of_range = out_of_range.any()
            except AttributeError:
                pass

            if out_of_range:
                raise TypeError("Data type conversion would lose information.")
    if numba_cuda.is_cuda_array(X):
        arr = cp.asarray(X, dtype=to_dtype)
        if legacy:
            return numba_cuda.as_cuda_array(arr)
        else:
            return CumlArray(data=arr)

    try:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            # TODO: Drop this pandas 2 branch once pandas 2 support is removed.
            if PANDAS_VERSION < Version("3.0"):
                return X.astype(to_dtype, copy=None)
            return X.astype(to_dtype)
        return X.astype(to_dtype, copy=False)
    except AttributeError:
        raise TypeError("Received unsupported input type: %s" % type(X))


def order_to_str(order):
    if order == "F":
        return "column ('F')"
    elif order == "C":
        return "row ('C')"


def sparse_scipy_to_cp(sp, dtype):
    """
    Convert object of scipy.sparse to
    cupyx.scipy.sparse.coo_matrix
    """

    coo = sp.tocoo()
    values = coo.data

    r = cp.asarray(coo.row)
    c = cp.asarray(coo.col)
    v = cp.asarray(values, dtype=dtype)

    return cupyx.scipy.sparse.coo_matrix((v, (r, c)), sp.shape)
