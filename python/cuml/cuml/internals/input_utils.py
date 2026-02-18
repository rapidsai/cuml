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
import scipy.sparse
from sklearn.utils._tags import get_tags

import cuml.internals.nvtx as nvtx
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType

global_settings = GlobalSettings()

cuml_array = namedtuple("cuml_array", "array n_rows n_cols dtype")

_input_type_to_str = {
    CumlArray: "cuml",
    SparseCumlArray: "cuml",
    np.ndarray: "numpy",
    pd.Series: "pandas",
    pd.DataFrame: "pandas",
    pd.Index: "pandas",
    cp.ndarray: "cupy",
    cudf.Series: "cudf",
    cudf.DataFrame: "cudf",
    cudf.Index: "cudf",
    numba_cuda.devicearray.DeviceNDArrayBase: "numba",
    cupyx.scipy.sparse.spmatrix: "cupy",
    scipy.sparse.spmatrix: "numpy",
    scipy.sparse.sparray: "numpy",
}

_input_type_to_mem_type = {
    np.ndarray: MemoryType.host,
    pd.Series: MemoryType.host,
    pd.DataFrame: MemoryType.host,
    scipy.sparse.spmatrix: MemoryType.host,
    scipy.sparse.sparray: MemoryType.host,
    cp.ndarray: MemoryType.device,
    cudf.Series: MemoryType.device,
    cudf.DataFrame: MemoryType.device,
    numba_cuda.devicearray.DeviceNDArrayBase: MemoryType.device,
    cupyx.scipy.sparse.spmatrix: MemoryType.device,
}

_SPARSE_TYPES = [
    SparseCumlArray,
    cupyx.scipy.sparse.spmatrix,
    scipy.sparse.spmatrix,
    scipy.sparse.sparray,
]


def get_supported_input_type(X):
    """
    Determines if the input object is a supported input array-like object or
    not. If supported, the type is returned. Otherwise, `None` is returned.

    Parameters
    ----------
    X : object
        Input object to test

    Notes
    -----
    To closely match the functionality of
    :func:`~cuml.internals.input_utils.input_to_cuml_array`, this method will
    return `cupy.ndarray` for any object supporting
    `__cuda_array_interface__` and `numpy.ndarray` for any object supporting
    `__array_interface__`.

    Returns
    -------
    array-like type or None
        If the array-like object is supported, the type is returned.
        Otherwise, `None` is returned.
    """
    # Check CumlArray first to shorten search time
    if isinstance(X, CumlArray):
        return CumlArray

    if isinstance(X, SparseCumlArray):
        return SparseCumlArray

    if isinstance(X, cudf.Series):
        if X.null_count != 0:
            return None
        else:
            return cudf.Series

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame

    if isinstance(X, pd.Series):
        return pd.Series

    if isinstance(X, pd.Index):
        return pd.Index

    if isinstance(X, cudf.DataFrame):
        return cudf.DataFrame

    if isinstance(X, cudf.Index):
        return cudf.Index

    # A cudf.pandas wrapped Numpy array defines `__cuda_array_interface__`
    # which means without this we'd always return a cupy array. We don't want
    # to match wrapped cupy arrays, they get dealt with later
    if getattr(X, "_fsproxy_slow_type", None) is np.ndarray:
        return np.ndarray

    if numba_cuda.devicearray.is_cuda_ndarray(X):
        return numba_cuda.devicearray.DeviceNDArrayBase

    if hasattr(X, "__cuda_array_interface__"):
        return cp.ndarray

    if hasattr(X, "__array_interface__"):
        # For some reason, numpy scalar types also implement
        # `__array_interface__`. See numpy.generic.__doc__. Exclude those types
        # as well as np.dtypes
        if not isinstance(X, np.generic) and not isinstance(X, type):
            return np.ndarray

    if cupyx.scipy.sparse.issparse(X):
        return cupyx.scipy.sparse.spmatrix

    if scipy.sparse.isspmatrix(X):
        return scipy.sparse.spmatrix

    if scipy.sparse.issparse(X) and X.ndim == 2:
        return scipy.sparse.sparray

    # Return None if this type is not supported
    return None


def determine_array_type(X):
    if X is None:
        return None

    # Get the generic type
    gen_type = get_supported_input_type(X)

    return _input_type_to_str.get(gen_type, None)


def determine_df_obj_type(X):
    if X is None:
        return None

    # Get the generic type
    gen_type = get_supported_input_type(X)

    if gen_type in (cudf.DataFrame, pd.DataFrame):
        return "dataframe"
    elif gen_type in (cudf.Series, pd.Series):
        return "series"

    return None


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

    return dtype


def determine_array_type_full(X):
    """
    Returns a tuple of the array type, and a boolean if it is sparse

    Parameters
    ----------
    X : array-like
        Input array to test

    Returns
    -------
    (string, bool) Returns a tuple of the array type string and a boolean if it
        is a sparse array.
    """
    if X is None:
        return None, None

    # Get the generic type
    gen_type = get_supported_input_type(X)

    if gen_type is None:
        return None, None

    return _input_type_to_str[gen_type], gen_type in _SPARSE_TYPES


def is_array_like(X, accept_lists=False):
    """Check if X is array-like.

    Parameters
    ----------
    X : object
        The object to check
    accept_lists : bool, default=False
        If True, treat list and tuple objects as array-like.
        If False, only treat actual array-like objects as array-like.

    Returns
    -------
    bool
        True if X is array-like, False otherwise
    """
    if (
        hasattr(X, "__cuda_array_interface__")
        or (
            hasattr(X, "__array_interface__")
            and not (isinstance(X, np.generic) or isinstance(X, type))
        )
        or isinstance(
            X,
            (
                SparseCumlArray,
                cudf.Series,
                pd.Series,
                cudf.DataFrame,
                pd.DataFrame,
            ),
        )
        or (accept_lists and isinstance(X, (list, tuple)))
    ):
        return True

    if cupyx.scipy.sparse.issparse(X):
        return True
    if scipy.sparse.issparse(X):
        return True
    if numba_cuda.devicearray.is_cuda_ndarray(X):
        return True
    return False


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
    ensure_2d=False,
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
    if ensure_2d:
        if X.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )

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


def validate_data(
    _estimator,
    /,
    X="no_validation",
    y="no_validation",
    *,
    reset=True,
    ensure_2d=True,
    accept_sparse=False,
    validate_separately=False,
    order="F",
    check_dtype=False,
    convert_to_dtype=False,
    convert_to_mem_type="device",
    check_cols=False,
    check_rows=False,
    force_contiguous=True,
):
    """Validate input data and manage ``n_features_in_``.

    Wraps :func:`input_to_cuml_array` with sklearn-compatible validation:

    * Rejects ``y=None`` for supervised estimators (tag-driven).
    * Enforces 2-D ``X`` by default.
    * Sets ``n_features_in_`` on fit (``reset=True``) and validates it on
      predict / transform (``reset=False``).

    Parameters
    ----------
    _estimator : cuml.internals.base.Base
        The estimator instance.  Positional-only.
    X : array-like or ``"no_validation"``
        Feature matrix.  ``"no_validation"`` skips X validation.
    y : array-like, None, or ``"no_validation"``
        Target array.  ``"no_validation"`` or ``None`` skips y validation.
        ``None`` is rejected when the estimator's tags indicate that y is
        required (e.g. classifiers and regressors).
    reset : bool, default=True
        If True, set ``n_features_in_`` from X.  If False, validate that X
        has the expected number of features.
    ensure_2d : bool, default=True
        Require X to be 2-D.
    accept_sparse : bool, default=False
        If True, sparse X is passed through without conversion (the
        estimator is responsible for handling it).  The ``ensure_2d`` and
        ``n_features_in_`` checks are still applied.  If False (default),
        sparse X is rejected by :func:`input_to_cuml_array` with a
        ``TypeError``.
    validate_separately : False or tuple of two dicts, default=False
        When a ``(X_kwargs, y_kwargs)`` tuple is given, X and y are each
        validated with their own ``input_to_cuml_array`` keyword arguments.
        This is the mechanism for passing different dtype / order / shape
        requirements for X and y.
    order, check_dtype, convert_to_dtype, check_cols, check_rows,
    force_contiguous :
        Forwarded to :func:`input_to_cuml_array` for X (and for y when
        ``validate_separately`` is False).

    Returns
    -------
    out : cuml_array or (cuml_array, cuml_array)
        Validated X as a ``cuml_array`` namedtuple, or ``(X_out, y_out)``
        when y is provided.
    """
    tags = get_tags(_estimator)

    if y is None and tags.target_tags.required:
        raise ValueError(
            f"This {_estimator.__class__.__name__} estimator "
            "requires y to be passed, but the target y is None."
        )

    from cuml.common.sparse_utils import is_sparse as _is_sparse

    no_val_X = isinstance(X, str) and X == "no_validation"
    no_val_y = y is None or (isinstance(y, str) and y == "no_validation")
    X_is_sparse = not no_val_X and accept_sparse and _is_sparse(X)

    if validate_separately:
        if no_val_X or no_val_y:
            raise ValueError(
                "validate_separately requires both X and y to be provided."
            )
        X_kwargs, y_kwargs = validate_separately
        X_kwargs.setdefault("ensure_2d", ensure_2d)
        X_out = input_to_cuml_array(X, **X_kwargs)
        y_kwargs.setdefault("ensure_2d", False)
        y_kwargs.setdefault("check_rows", X_out.n_rows)
        y_out = input_to_cuml_array(y, **y_kwargs)
    else:
        if not no_val_X:
            if X_is_sparse:
                # Sparse pass-through: skip input_to_cuml_array conversion
                # but still enforce ensure_2d.
                if ensure_2d:
                    if X.ndim == 1:
                        raise ValueError(
                            "Expected 2D array, got 1D array instead.\n"
                            "Reshape your data either using "
                            "array.reshape(-1, 1) if your data has a "
                            "single feature or array.reshape(1, -1) if "
                            "it contains a single sample."
                        )
                n_rows, n_cols = X.shape
                X_out = cuml_array(
                    array=X,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    dtype=X.dtype,
                )
            else:
                X_out = input_to_cuml_array(
                    X,
                    order=order,
                    check_dtype=check_dtype,
                    convert_to_dtype=convert_to_dtype,
                    convert_to_mem_type=convert_to_mem_type,
                    check_cols=check_cols,
                    check_rows=check_rows,
                    force_contiguous=force_contiguous,
                    ensure_2d=ensure_2d,
                )

        if not no_val_y:
            y_kwargs = dict(
                ensure_2d=False,
                force_contiguous=force_contiguous,
            )
            if not no_val_X:
                y_kwargs["check_rows"] = X_out.n_rows
                # When dtype conversion was requested for X, also convert y
                # to X's resulting dtype.  This ensures regressors get
                # matching float dtypes for X and y.
                if convert_to_dtype and not X_is_sparse:
                    y_kwargs["convert_to_dtype"] = X_out.dtype
            y_out = input_to_cuml_array(y, **y_kwargs)

    # n_features_in_ management
    if not no_val_X and ensure_2d:
        if reset:
            _estimator.n_features_in_ = X_out.n_cols
        elif hasattr(_estimator, "n_features_in_"):
            if X_out.n_cols != _estimator.n_features_in_:
                raise ValueError(
                    f"X has {X_out.n_cols} features, but "
                    f"{_estimator.__class__.__name__} is expecting "
                    f"{_estimator.n_features_in_} features as input."
                )

    if no_val_y:
        return X_out
    return X_out, y_out


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
    ensure_2d=False,
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
        ensure_2d=ensure_2d,
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


def input_to_host_array_with_sparse_support(X):
    if X is None:
        return None
    if scipy.sparse.issparse(X):
        return X
    _array_type, is_sparse = determine_array_type_full(X)
    if is_sparse:
        if _array_type == "cupy":
            return SparseCumlArray(X).to_output(output_type="scipy")
        elif _array_type == "cuml":
            return X.to_output(output_type="scipy")
        elif _array_type == "numpy":
            return X
        else:
            raise ValueError(f"Unsupported sparse array type: {_array_type}.")
    return input_to_host_array(X).array


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
