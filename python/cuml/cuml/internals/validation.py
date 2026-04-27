#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numbers
import warnings

import cudf
import cudf.pandas
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted

__all__ = (
    "check_is_fitted",
    "check_random_seed",
    "check_features",
    "check_consistent_length",
    "check_all_finite",
    "check_non_negative",
    "check_array",
    "check_y",
    "check_sample_weight",
    "check_inputs",
)


def check_random_seed(random_state) -> int:
    """Turn a `random_state` argument into a seed.

    Parameters
    ----------
    random_state : None | int | instance of RandomState
        If random_state is None, return a random int as seed.
        If random_state is an int, return it.
        If random_state is a RandomState instance, derive a seed from it.

    Returns
    -------
    seed : int
        A seed in the range [0, 2**32 - 1].
    """
    if isinstance(random_state, numbers.Integral):
        if random_state < 0 or random_state >= 2**32:
            raise ValueError(
                f"Expected `0 <= random_state <= 2**32 - 1`, got {random_state}"
            )
        return int(random_state)

    if random_state is None:
        randint = np.random.randint
    elif isinstance(
        random_state, (np.random.RandomState, cp.random.RandomState)
    ):
        randint = random_state.randint
    else:
        raise TypeError(
            f"`random_state` must be an `int`, an instance of `RandomState`, or `None`. "
            f"Got {random_state!r} instead."
        )

    # randint returns in [low, high), so high=2**32 to sample all uint32s
    return int(randint(low=0, high=2**32, dtype=np.uint32))


def _check_shape(
    shape,
    *,
    ensure_2d=True,
    ensure_min_samples=0,
    ensure_min_features=0,
    array_type=None,
) -> None:
    """Check that an input shape is as expected.

    Parameters
    ----------
    shape : tuple
        The array shape.
    ensure_2d : bool, default=True
        If True, only 2D arrays are accepted. Otherwise accepts 1D or 2D
        arrays.
    ensure_min_samples : int, default=0
        A minimum number of samples to require. Defaults to 0 for no minimum.
    ensure_min_features : int, default=0
        A minimum number of features to require. Defaults to 0 for no minimum.
    array_type : type or None, default=None
        The type of the array-like object. Used in error messages.
    """
    ndim = len(shape)

    if ndim == 0 or ndim == 1 and ensure_2d:
        if issubclass(array_type, (cudf.Series, pd.Series)):
            msg = (
                f"Expected a 2-dimensional container but got {array_type} "
                "instead. Pass a DataFrame containing a single row (i.e. "
                "single sample) or a single column (i.e. single feature) "
                "instead."
            )
        else:
            kind = "scalar" if ndim == 0 else "1D"
            msg = (
                f"Expected 2D array, got {kind} array instead. Reshape your data "
                "using array.reshape(-1, 1) if your data has a single feature, "
                "or array.reshape(1, -1) if it contains a single sample."
            )
        raise ValueError(msg)
    elif ndim > 2:
        raise ValueError(f"Expected 2D array, got {ndim}D array instead.")

    if ensure_min_samples > 0:
        n_samples = shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s) (shape={shape}) "
                f"while a minimum of {ensure_min_samples} is required."
            )

    if ensure_min_features > 0 and ndim == 2:
        n_features = shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                f"Found array with {n_features} feature(s) (shape={shape}) "
                f"while a minimum of {ensure_min_features} is required."
            )


def _get_n_features(X):
    """Get the number of features in X."""
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return 0
        row = X[0]
        # For non-array inputs, we assume that all nested lists have the same
        # length. This matches sklearn's implementation as well. If this
        # assumption isn't true, then later validation code will error anyway.
        # We only take the length of sub-sequences that numpy wouldn't treat as
        # single elements.
        if not isinstance(row, (str, bytes, dict)):
            try:
                return len(row)
            except Exception:
                pass

    if hasattr(X, "shape"):
        shape = X.shape
    elif hasattr(X, "__cuda_array_interface__"):
        shape = X.__cuda_array_interface__["shape"]
    elif hasattr(X, "__array_interface__"):
        shape = X.__array_interface__["shape"]
    else:
        shape = np.asarray(X).shape

    _check_shape(shape, ensure_2d=True, array_type=type(X))

    return shape[1]


def _get_n_samples(X):
    """Get the number of samples in X."""

    if (shape := getattr(X, "shape", None)) is not None:
        if len(shape) == 0:
            raise TypeError("Expected array-like, got scalar instead.")
        return shape[0]

    try:
        return len(X)
    except TypeError as exc:
        raise TypeError(
            f"Expected array-like, got {type(X)} instead."
        ) from exc


def _get_feature_names(X):
    """Get feature names from X.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    if isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        feature_names = np.asarray(
            list(X.__dataframe__().column_names()), dtype=object
        )
    else:
        return None

    if len(feature_names) == 0:
        # No features, just return None
        return None

    # Check the types of the column names.
    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
    if len(types) == 1 and types[0] == "str":
        return feature_names
    elif len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    return None


def check_features(estimator, X, reset=False) -> None:
    """Check or set ``n_features_in_`` and ``feature_names_in_``.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The original user-provided `X` input. No conversion or processing steps
        should have occurred to this array yet.
    reset : bool, default=False
        If True, ``n_features_in_`` and ``feature_names_in_`` are set on
        ``estimator`` to match ``X``. Otherwise ``X`` is checked to match the
        existing ``n_features_in_`` and ``feature_names_in_``. ``reset=True``
        should be used for fit-like methods, and False otherwise.
    """
    n_features = _get_n_features(X)
    feature_names = _get_feature_names(X)

    if reset:
        estimator.n_features_in_ = n_features
        if feature_names is not None:
            estimator.feature_names_in_ = feature_names
        elif hasattr(estimator, "feature_names_in_"):
            # Clear old feature names if present
            delattr(estimator, "feature_names_in_")
        return

    est_feature_names = getattr(estimator, "feature_names_in_", None)

    # Check feature_names_in_ first
    if est_feature_names is not None or feature_names is not None:
        if est_feature_names is None:
            warnings.warn(
                f"X has feature names, but {estimator.__class__.__name__} was fitted "
                "without feature names"
            )

        elif feature_names is None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {estimator.__class__.__name__} was fitted with feature names"
            )

        elif len(est_feature_names) != len(feature_names) or np.any(
            est_feature_names != feature_names
        ):
            unexpected = sorted(
                set(feature_names).difference(est_feature_names)
            )
            missing = sorted(set(est_feature_names).difference(feature_names))

            parts = [
                "The feature names should match those that were passed during fit."
            ]
            for heading, names in [
                ("Feature names unseen at fit time:", unexpected),
                ("Feature names seen at fit time, yet now missing:", missing),
            ]:
                if names:
                    parts.append(heading)
                    parts.extend([f"- {name}" for name in names[:5]])
                    if len(names) > 5:
                        parts.append("- ...")

            if not missing and not unexpected:
                parts.append(
                    "Feature names must be in the same order as they were in fit."
                )

            msg = "\n".join(parts)
            raise ValueError(msg)

    # Then check n_features_in_
    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )


def check_consistent_length(*arrays) -> None:
    """Check whether all inputs have the same number of samples.

    Typically should be called after arrays have been validated and normalized
    by other checks, but also works on typical unvetted user inputs.

    Parameters
    ----------
    *arrays : array or None
        The input variables to validate. None-values are ignored.
    """
    lengths = [_get_n_samples(X) for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Found input variables with inconsistent number of samples: "
            f"{sorted(n for n in lengths)}"
        )


# Returns status in a bitfield:
# 0b01: contains NaN
# 0b10: contains +/-inf
_cupy_any_inf_or_nan = cp.ReductionKernel(
    "T x",
    "uint8 out",
    "((unsigned char)isinf(x)) << 1 | ((unsigned char)isnan(x))",
    "a | b",
    "out = a",
    "0",
    "any_inf_or_nan",
)


def check_all_finite(array, *, allow_nan=False, input_name=None) -> None:
    """Check if all input values are finite.

    This check is skipped if scikit-learn's ``assume_finite`` option is
    configured via ``sklearn.set_config(assume_finite=True)``.

    Parameters
    ----------
    array : dense or sparse array
        The array to check.
    allow_nan : bool, default=False
        Whether to allow NaN values.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    """
    if not np.isdtype(array.dtype, "real floating"):
        # No-op for non floating inputs
        return

    if sklearn.get_config()["assume_finite"]:
        # no-op if assume_finite configured
        return

    if cp_sp.issparse(array) or sp.issparse(array):
        array = array.data

    if not array.size:
        # No-op for empty inputs
        return

    has_nan = has_inf = False
    if isinstance(array, cp.ndarray):
        status = _cupy_any_inf_or_nan(array).item()
        has_nan = status & 0b01
        has_inf = status & 0b10
    else:
        # First try an O(1) space solution for the common case
        with np.errstate(over="ignore", invalid="ignore"):
            x_sum = array.sum()
        if not np.isfinite(x_sum):
            # We can't infer anything from the value of x_sum being non-finite
            # - NaN could mean NaN present, or both -inf and inf
            # - inf could mean inf present, or just overflow
            # Here we selectively apply O(n) space fallbacks as needed.
            if allow_nan:
                has_inf = np.isinf(array).any()
            elif not (has_nan := np.isnan(array).any()):
                has_inf = np.isinf(array).any()

    if has_nan and not allow_nan:
        msg = "NaN"
    elif has_inf:
        msg = f"infinity or a value too large for {array.dtype!r}"
    else:
        msg = None

    if msg is not None:
        raise ValueError(f"Input {input_name or 'array'} contains {msg}.")


def check_non_negative(array, *, input_name=None) -> None:
    """Check if all input values are non-negative.

    Parameters
    ----------
    array : dense or sparse array
        The array to check.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    """
    if cp_sp.issparse(array) or sp.issparse(array):
        array = array.data
    xp = cp if isinstance(array, cp.ndarray) else np
    if array.size != 0 and xp.nanmin(array) < 0:
        suffix = f" passed to {input_name}" if input_name is not None else ""
        raise ValueError(f"Negative values in data{suffix}")


def _ensure_int32_sparse(array):
    """Convert sparse array to int32 indices if possible, and error otherwise"""
    INT32_MAX = (1 << 31) - 1

    # All sparse arrays must have shapes and nnz that fit in an int32. In addition,
    # CSR, CSC, and BSR must have indices/indptr that fit in an int32.
    if (
        any(s > INT32_MAX for s in array.shape)
        or array.nnz > INT32_MAX
        or (
            array.format in ["csr", "csc", "bsr"]
            and (
                len(array.indices) > INT32_MAX or len(array.indptr) > INT32_MAX
            )
        )
    ):
        raise ValueError(
            "Only sparse matrices with int32 indices are currently supported."
        )

    # Definitely safe to downscast to int32, but only cast if needed since
    # the sparse constructors do a little bit of work.
    if array.format == "coo":
        if array.row.dtype == "int32" and array.col.dtype == "int32":
            return array
        return type(array)(
            (
                array.data,
                (
                    array.row.astype("int32", copy=False),
                    array.col.astype("int32", copy=False),
                ),
            ),
            shape=array.shape,
        )
    elif array.format == "dia":
        if array.offsets.dtype == "int32":
            return array
        return type(array)(
            (array.data, array.offsets.astype("int32")), shape=array.shape
        )
    elif array.format in ["csr", "csc", "bsr"]:
        if array.indices.dtype == "int32" and array.indptr.dtype == "int32":
            return array
        return type(array)(
            (
                array.data,
                array.indices.astype("int32", copy=False),
                array.indptr.astype("int32", copy=False),
            ),
            shape=array.shape,
        )
    else:
        # Other type without numeric indices, can just return
        return array


def check_array(
    array,
    *,
    accept_sparse=False,
    accept_large_sparse=False,
    dtype=None,
    convert_dtype=True,
    mem_type="device",
    order="A",
    ensure_all_finite=True,
    ensure_non_negative=False,
    ensure_2d=True,
    ensure_min_samples=1,
    ensure_min_features=1,
    input_name=None,
    return_index=False,
):
    """Validate and coerce an array-like to a supported type.

    Parameters
    ----------
    array : array-like
        The array-like input to validate.
    accept_sparse : bool, str, list[str], default=False
        The sparse matrix format(s) to support. If the input is sparse
        but not in a supported format, it will be converted to the first
        listed format. Pass True to support any input format. The default
        of False will raise an error on sparse inputs.
    accept_large_sparse : bool, default=False
        Whether large (int64) indices are supported for sparse containers with
        CSR/CSC/COO/BSR formats. If not supported, an appropriate error will be
        raised if the sparse indices aren't int32.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    convert_dtype : bool, default=True
        Whether to support dtype conversion. If False, an error will be raised
        if the input isn't a supported dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray`` if dense, or a ``cupyx.scipy.sparse.spmatrix`` if
        sparse. If 'host', the output will be a ``numpy.ndarray`` if dense, or
        a ``scipy.sparse.spmatrix`` if sparse. If ``None``, the output will
        have the same memory type as the input (i.e. device if already on
        device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    ensure_all_finite : bool or 'allow-nan', default=True
        If True, an error will be raised if non-finite values are found in the
        input. If 'allow-nan', an error will be raised if infinite values are
        found (but not for NaN). If False then ``check_all_finite`` is skipped.
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in the
        input. By default ``check_non_negative`` is skipped.
    ensure_2d : bool, default=True
        If True, the input must be 2D. If False, 1D or 2D inputs are accepted.
    ensure_min_samples : int, default=1
        A minimum number of samples to require. Set to 0 for no minimum.
    ensure_min_features : int, default=1
        A minimum number of features to require for 2D inputs. Set to 0 for no
        minimum.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    return_index : bool, default=False
        Whether to return the index of ``array`` (if a dataframe-like value).
        This is useful for functions that need to return an output with a
        dataframe index aligned with the input.

    Returns
    -------
    array : dense or sparse array
        The converted and validated array. Depending on input and parameters,
        will be one of ``cupy.ndarray``, ``numpy.ndarray``,
        ``cupyx.scipy.sparse.spmatrix``, or ``scipy.sparse.spmatrix``.
    index : pandas.Index, cudf.Index, or None
        The index of the input if a dataframe-like, or None if no index. The
        index will be converted to match ``mem_type``. Only returned if
        ``return_index=True``.
    """
    # Normalize and validate arguments
    if mem_type not in ("device", "host", None):
        raise ValueError(f"Unsupported {mem_type=!r}")
    if order not in ("F", "C", "A", None):
        raise ValueError(f"Unsupported {order=!r}")
    if dtype is not None:
        if not isinstance(dtype, (list, tuple)):
            dtype = [dtype]
        dtype = [np.dtype(i) for i in dtype]

    # Extract original array type and dtype (when possible)
    array_type = type(array)
    if isinstance(array, (cudf.DataFrame, pd.DataFrame)):
        if all(isinstance(dt, np.dtype) for dt in array.dtypes):
            array_dtype = np.result_type(*array.dtypes)
        elif any(dt == "object" for dt in array.dtypes):
            array_dtype = np.dtype("object")
        else:
            array_dtype = None
    else:
        array_dtype = getattr(array, "dtype", None)
        if not isinstance(array_dtype, np.dtype):
            array_dtype = None

    # Infer proper output dtype
    if array_dtype is not None:
        # Check for complex inputs before conversion when possible
        if np.isdtype(array_dtype, "complex floating"):
            raise ValueError("Complex data not supported")
        if dtype is None:
            dtype = array_dtype
        elif array_dtype not in dtype:
            if convert_dtype:
                # Convert to first provided dtype
                dtype = dtype[0]
            else:
                raise ValueError(
                    f"Expected array with dtype in {[str(d) for d in dtype]} "
                    f"but got {str(array_dtype)!r}"
                )
        else:
            dtype = array_dtype
    elif dtype is not None:
        # No original dtype, use first dtype in list inputs
        dtype = dtype[0]

    # Coerce `array` to numpy/cupy/scipy.sparse/cupyx.scipy.sparse values as
    # requested. For dataframe-like inputs also extract the index for later use.
    index = None
    if cp_sp.issparse(array) or sp.issparse(array):
        # Handle sparse inputs
        if isinstance(accept_sparse, str):
            accept_sparse = [accept_sparse]
        elif accept_sparse is True:
            # Only support formats cupyx.scipy.sparse supports
            accept_sparse = ["csr", "coo", "csc", "dia"]

        if not accept_sparse:
            padded_input = f" for {input_name}" if input_name else ""
            raise TypeError(
                f"Sparse data was passed{padded_input}, but dense data is required. "
                "Use '.toarray()' to convert to a dense array."
            )
        # Coerce to accepted format if needed
        if array.format not in accept_sparse:
            array = array.asformat(accept_sparse[0])
        if not accept_large_sparse:
            array = _ensure_int32_sparse(array)

        # Validate dimensions and shape are as expected. We do this here
        # _before_ host/device conversion, since cupyx doesn't have a sparse
        # array type and thus only supports 2D inputs.
        _check_shape(
            array.shape,
            array_type=array_type,
            ensure_2d=ensure_2d,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
        )

        # Coerce data to accepted dtype if needed
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype)

        # Coerce to device or host if needed
        if mem_type == "device" and not cp_sp.issparse(array):
            if array.ndim != 2:
                raise ValueError("cupyx.scipy.sparse only supports 2D arrays")
            array = getattr(cp_sp, f"{array.format}_matrix")(array)
        elif mem_type == "host" and not sp.issparse(array):
            array = array.get()
    else:
        # Handle dense inputs
        if isinstance(array, (cudf.DataFrame, cudf.Series)):
            # Handle cudf inputs
            index = array.index
            if mem_type == "host":
                array = np.asarray(
                    array.to_numpy(dtype=dtype), dtype=dtype, order=order
                )
            else:
                # XXX: the dtype keyword to `to_cupy` is buggy, and also
                # doesn't support all dtype coercions. For now we do a
                # manual cast to handle any coercions.
                # See https://github.com/rapidsai/cudf/issues/22136.
                if dtype is not None:
                    array = array.astype(dtype, copy=False)
                array = cp.asarray(array.to_cupy(), dtype=dtype, order=order)
        elif isinstance(array, (pd.DataFrame, pd.Series)):
            # Handle pandas inputs
            index = array.index
            array = array.to_numpy(dtype=dtype)
            if mem_type == "device":
                array = cp.asarray(array, dtype=dtype, order=order)
            elif (
                mem_type is None
                and cudf.pandas.LOADED
                and np.isdtype(array.dtype, ("numeric", "bool"))
            ):
                # We treat pandas objects with supported dtypes as device
                # memory when running under cudf.pandas. Note that the output
                # of `to_numpy` in cudf.pandas returns a proxy array that's
                # remains on device, so we're not paying a device<>host
                # roundtrip cost here.
                array = cp.asarray(array, dtype=dtype, order=order)
            else:
                array = np.asarray(array, dtype=dtype, order=order)
        elif hasattr(array, "__cuda_array_interface__"):
            # Handle device-backed array-like inputs
            if mem_type == "host":
                array = cp.asnumpy(array, order=order or "A")
                # Possible 2nd copy done on host for dtype enforcement
                array = np.asarray(array, dtype=dtype, order=order)
            else:
                array = cp.asarray(array, dtype=dtype, order=order)
        else:
            # Handle all other inputs
            if mem_type == "device":
                array = cp.asarray(array, dtype=dtype, order=order)
            else:
                array = np.asarray(array, dtype=dtype, order=order)

        # XXX: order="A" isn't consistently handled by cupy or numpy. If a copy
        # was made, the output will definitely already be contiguous. If no
        # copy was already made though, we may need to make one to enforce
        # contiguity (here we default to F-contiguous, mirroring what _most_
        # code paths do with `order="A"`).
        if order == "A" and not (
            array.flags["F_CONTIGUOUS"] or array.flags["C_CONTIGUOUS"]
        ):
            array = (
                cp.asarray(array, order="F")
                if isinstance(array, cp.ndarray)
                else np.asarray(array, order="F")
            )

        # Validate dimensions and shape are as expected
        _check_shape(
            array.shape,
            array_type=array_type,
            ensure_2d=ensure_2d,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
        )

    # Check for complex inputs after conversion for cases when `dtype=None`
    if np.isdtype(array.dtype, "complex floating"):
        raise ValueError("Complex data not supported")

    # Validate data meets expected value requirements
    if ensure_all_finite:
        check_all_finite(
            array,
            allow_nan=ensure_all_finite == "allow-nan",
            input_name=input_name,
        )
    if ensure_non_negative:
        check_non_negative(array, input_name=input_name)

    # Process index if requested, then return
    if return_index:
        if isinstance(index, cudf.Index) and mem_type == "host":
            index = (
                cudf.pandas.as_proxy_object(index)
                if cudf.pandas.LOADED
                else index.to_pandas()
            )
        elif isinstance(index, pd.Index) and mem_type == "device":
            index = cudf.Index(index)
        return array, index
    else:
        return array


# Returns status in a bitfield:
# 0b001: contains NaN
# 0b010: contains +/-inf
# 0b100: contains real (non-integral) values
_cupy_any_inf_or_nan_or_real = cp.ReductionKernel(
    "T x",
    "uint8 out",
    (
        "((unsigned char)(ceil(x) != x)) << 2 "
        "| ((unsigned char)isinf(x)) << 1 "
        "| ((unsigned char)isnan(x))"
    ),
    "a | b",
    "out = a",
    "0",
    "any_inf_or_nan_or_real",
)


def _check_classification_targets(y):
    """Check if `y` is composed of valid class labels.

    Catches NaN, infinity, and non-integral inputs.

    Parameters
    ----------
    y : cupy.ndarray
        The ``y`` input to check.
    """
    if y.dtype.kind == "f":
        status = _cupy_any_inf_or_nan_or_real(y)
        if status & 0b001:
            raise ValueError("Input y contains NaN.")
        elif status & 0b010:
            raise ValueError(
                f"Input y contains infinity or a value too large for {y.dtype!r}."
            )
        elif status & 0b100:
            raise ValueError(
                "Unknown label type: continuous. Maybe you are trying to fit a "
                "classifier, which expects discrete classes on a regression target "
                "with continuous values."
            )


def check_y(
    y,
    *,
    dtype=None,
    convert_dtype=True,
    mem_type="device",
    order="A",
    accept_multi_output=False,
    return_classes=False,
):
    """Validate and coerce ``y`` to a supported type.

    Parameters
    ----------
    y : array-like
        The array-like input to validate.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype enforcement is performed;
        for classifiers the output will be a suitable integral type, otherwise
        the input dtype will be used. Pass a dtype or a list of supported
        dtypes to enforce a dtype for the output. If the input doesn't have a
        supported dtype, it will be converted to the first listed dtype.
    convert_dtype : bool, default=True
        Whether to support dtype conversion. If False, an error will be raised
        if the input isn't a supported dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray``. If 'host', the output will be a ``numpy.ndarray``. If
        ``None``, the output will have the same memory type as the input (i.e.
        device if already on device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    accept_multi_output : bool, default=False
        Whether multi-output y is accepted. By default only 1D inputs (or 2D
        inputs with a single column) are accepted. Set to True to accept
        multi-column inputs as well.
    return_classes : bool, default=False
        Set to True to also label encode ``y`` and return the ``classes``.

    Returns
    -------
    y : cupy.ndarray or numpy.ndarray
        The converted and validated array.
    classes : numpy.ndarray or list[numpy.ndarray]
        The collected classes for a classifier input. Only returned if
        ``return_classes=True``.
    """
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None"
        )

    # Normalize `dtype` arg
    if dtype is not None:
        if not isinstance(dtype, (list, tuple)):
            dtype = [dtype]
        dtype = [np.dtype(i) for i in dtype]

    # Coerce `y` to a supported array type
    if return_classes:
        # cudf may coerce the dtype, store the original so we can cast back later
        input_dtype = y.dtype if isinstance(y, np.ndarray) else None

        # No cuda container supports all dtypes. Here we coerce to cupy when
        # possible, falling back to cudf Series/DataFrame otherwise.
        if not isinstance(y, (cudf.DataFrame, cudf.Series)):
            y = check_array(
                y,
                mem_type=None,
                ensure_2d=False,
                ensure_min_samples=0,
                ensure_all_finite=False,
                input_name="y",
            )
            # If no original dtype found on input, use the coerced one instead
            if input_dtype is None:
                input_dtype = y.dtype
            if mem_type is None:
                mem_type = "host" if isinstance(y, np.ndarray) else "device"
            if np.isdtype(y.dtype, ("numeric", "bool")):
                y = cp.asarray(y)
            elif (
                y.dtype == "object"
                and y.size
                and not isinstance(y.flat[0], str)
            ):
                raise ValueError(
                    "Unknown label type: unknown. Maybe you are trying to fit a "
                    "classifier, which expects discrete classes on a regression target "
                    "with continuous values."
                )
            else:
                y = (cudf.DataFrame if y.ndim == 2 else cudf.Series)(
                    y, dtype=(np.dtype("O") if y.dtype.kind in "U" else None)
                )
        elif mem_type is None:
            mem_type = "device"
    else:
        y = check_array(
            y,
            dtype=dtype,
            convert_dtype=convert_dtype,
            mem_type=mem_type,
            order=order,
            ensure_2d=False,
            ensure_min_samples=0,
            input_name="y",
        )

    # Warn/error appropriately for 2D inputs
    if y.ndim == 2:
        if y.shape[1] == 1 and (return_classes or not accept_multi_output):
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example "
                "using ravel().",
                DataConversionWarning,
            )
            if return_classes:
                y = (
                    y.iloc[:, 0]
                    if isinstance(y, cudf.DataFrame)
                    else y.ravel()
                )
        elif not accept_multi_output:
            raise ValueError(
                f"y should be a 1d array, got an array of shape {y.shape} instead."
            )

    if not return_classes:
        return y

    # For classifiers, we label encode y and return the integral labels as well
    # as the classes.
    def _encode(y):
        """Encode `y` to codes and classes"""
        _check_classification_targets(y)
        if isinstance(y, cudf.Series):
            y = y.astype("category")
            codes = cp.asarray(y.cat.codes)
            classes = y.cat.categories.to_numpy()
            # cudf will sometimes translate non-numeric dtypes. Coerce back to
            # the input dtype if the input was originally a numpy array.
            if input_dtype is not None:
                classes = classes.astype(input_dtype, copy=False)
        else:
            classes, codes = cp.unique(y, return_inverse=True)
            classes = classes.get()
        return codes, classes

    # Return C order if C requested, otherwise F.
    if order != "C":
        order = "F"

    if y.ndim == 1:
        y, classes = _encode(y)
        if dtype is not None and y.dtype not in dtype:
            y = y.astype(dtype[0])
    else:
        getter = y.iloc if isinstance(y, cudf.DataFrame) else y
        encoded_cols, classes = zip(
            *(_encode(getter[:, i]) for i in range(y.shape[1]))
        )
        classes = list(classes)
        # Infer output dtype
        out_dtype = cp.result_type(*(c.dtype for c in encoded_cols))
        if dtype is not None and out_dtype not in dtype:
            dtype = dtype[0]
        else:
            dtype = out_dtype
        y = cp.empty(shape=y.shape, dtype=dtype, order=order)
        for i, col in enumerate(encoded_cols):
            y[:, i] = col

    if mem_type == "host":
        # convert back to host if needed
        y = y.get(order=order)

    return y, classes


def check_sample_weight(
    sample_weight,
    *,
    dtype=None,
    convert_dtype=True,
    mem_type="device",
    order="A",
    ensure_non_negative=False,
):
    """Validate and coerce ``sample_weight`` to a supported type.

    Parameters
    ----------
    sample_weight : array-like, scalar, or None
        The ``sample_weight`` input to validate.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    convert_dtype : bool, default=True
        Whether to support dtype conversion. If False, an error will be raised
        if the input isn't a supported dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray``. If 'host', the output will be a ``numpy.ndarray``. If
        ``None``, the output will have the same memory type as the input (i.e.
        device if already on device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in the
        input. By default ``check_non_negative`` is skipped.

    Returns
    -------
    sample_weight : cupy.ndarray, numpy.ndarray, or None
        The converted and validated weights.
    """
    if sample_weight is None:
        return None

    all_zero_msg = "Sample weights must contain at least one non-zero number."

    if np.isscalar(sample_weight):
        if sample_weight == 0:
            raise ValueError(all_zero_msg)
        elif ensure_non_negative and sample_weight < 0:
            raise ValueError("Negative values in data passed to sample_weight")
        elif np.isnan(sample_weight):
            raise ValueError("Input sample_weight contains NaN")
        elif np.isinf(sample_weight):
            raise ValueError("Input sample_weight contains infinity")
        else:
            # A uniform sample_weight is the same as unweighted
            return None

    sample_weight = check_array(
        sample_weight,
        dtype=dtype,
        convert_dtype=convert_dtype,
        mem_type=mem_type,
        order=order,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_non_negative=ensure_non_negative,
        input_name="sample_weight",
    )
    if sample_weight.ndim != 1:
        raise ValueError(
            f"Sample weights must be 1D array or scalar, got "
            f"{sample_weight.ndim}D array."
        )

    if sample_weight.size and (sample_weight == 0).all():
        raise ValueError(all_zero_msg)
    return sample_weight


def check_inputs(
    estimator,
    X,
    y=...,
    sample_weight=...,
    *,
    accept_sparse=False,
    accept_large_sparse=False,
    dtype=None,
    y_dtype=...,
    sample_weight_dtype=...,
    convert_dtype=True,
    mem_type="device",
    order="A",
    ensure_all_finite=True,
    ensure_non_negative=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    accept_multi_output=False,
    return_classes=False,
    return_index=False,
    reset=False,
):
    """Validate and coerce common inputs to an estimator method.

    This plumbs together several common checks. For a method with ``X``, ``y``,
    and ``sample_weight``, it's roughly equivalent to:

    ```
    check_features(estimator, X, reset=reset)
    X = check_array(X, input_name="X", ...)
    y = check_y(y, ...)
    sample_weight = check_sample_weight(sample_weight, ...)
    check_consistent_length(X, y, sample_weight)
    ```

    If this pattern doesn't work for an estimator, you can call always call
    some of the individual checks directly.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The ``X`` input.
    y : array-like, default=...
        The ``y`` input. May be omitted.
    sample_weight : array-like, scalar, or None
        The ``sample_weight`` input. May be omitted.
    accept_sparse : bool, str, list[str], default=False
        The sparse matrix format(s) to support. If the input is sparse
        but not in a supported format, it will be converted to the first
        listed format. Pass True to support any input format. The default
        of False will raise an error on sparse inputs.
    accept_large_sparse : bool, default=False
        Whether large (int64) indices are supported for sparse containers with
        CSR/CSC/COO/BSR formats. If not supported, an appropriate error will be
        raised if the sparse indices aren't int32.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support for X. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    y_dtype : None, dtype, list[dtype], default=...
        The dtype(s) to support for y. If not specified, defaults to
        the output dtype of ``X``.
    sample_weight_dtype : None, dtype, list[dtype], default=...
        The dtype(s) to support for sample_weight. If not specified, defaults
        to the output dtype of ``X``.
    convert_dtype : bool, default=True
        Whether to support dtype conversion. If False, an error will be raised
        if the input isn't a supported dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray`` if dense, or a ``cupyx.scipy.sparse.spmatrix`` if
        sparse. If 'host', the output will be a ``numpy.ndarray`` if dense, or
        a ``scipy.sparse.spmatrix`` if sparse. If ``None``, the output will
        have the same memory type as the input (i.e. device if already on
        device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    ensure_all_finite : bool or 'allow-nan', default=True
        If True, an error will be raised if non-finite values are found in X.
        If 'allow-nan', an error will be raised if infinite values are
        found (but not for NaN). If False then ``check_all_finite`` is skipped.
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in X. By
        default ``check_non_negative`` is skipped.
    ensure_min_samples : int, default=1
        A minimum number of samples to require. Set to 0 for no minimum.
    ensure_min_features : int, default=1
        A minimum number of features to require for 2D inputs. Set to 0 for no
        minimum.
    accept_multi_output : bool, default=False
        Whether multi-output y is accepted. By default only 1D inputs (or 2D
        inputs with a single column) are accepted. Set to True to accept
        multi-column inputs as well.
    return_classes : bool, default=False
        Set to True to also label encode ``y`` and return the ``classes``.
    return_index : bool, default=False
        Whether to return the index of ``X`` (if a dataframe-like value).
        This is useful for functions that need to return an output with a
        dataframe index aligned with the input.
    reset : bool, default=False
        If True, ``n_features_in_`` and ``feature_names_in_`` are set on
        ``estimator`` to match ``X``. Otherwise ``X`` is checked to match the
        existing ``n_features_in_`` and ``feature_names_in_``. ``reset=True``
        should be used for fit-like methods, and False otherwise.

    Returns
    -------
    X : dense or sparse array
        The converted and validated array. Depending on input and parameters,
        will be one of ``cupy.ndarray``, ``numpy.ndarray``,
        ``cupyx.scipy.sparse.spmatrix``, or ``scipy.sparse.spmatrix``.
    y : cupy.ndarray or numpy.ndarry
        The converted and validated array. Omitted if no ``y`` provided.
    sample_weight : cupy.ndarray, numpy.ndarray, or None
        The converted and validated weights. Omitted if no ``sample_weight``
        provided.
    classes : numpy.ndarray or list[numpy.ndarray]
        The collected classes from ``y`` for a classifier input. Only returned
        if ``return_classes=True``.
    index : pandas.Index, cudf.Index, or None
        The index of the input if a dataframe-like, or None if no index. The
        index will be converted to match ``mem_type``. Only returned if
        ``return_index=True``.
    """
    check_features(estimator, X, reset=reset)

    # Validate X
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        convert_dtype=convert_dtype,
        mem_type=mem_type,
        order=order,
        ensure_all_finite=ensure_all_finite,
        ensure_non_negative=ensure_non_negative,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        return_index=return_index,
        input_name="X",
    )
    if return_index:
        X, index = X
    else:
        index = None
    out = [X]

    # Validate y
    classes = None
    if y is not ...:
        if y_dtype is ...:
            y_dtype = X.dtype
        y = check_y(
            y,
            dtype=y_dtype,
            convert_dtype=convert_dtype,
            mem_type=mem_type,
            order=order,
            accept_multi_output=accept_multi_output,
            return_classes=return_classes,
        )
        if return_classes:
            y, classes = y
        out.append(y)

    # Validate sample_weight
    if sample_weight is not ...:
        if sample_weight_dtype is ...:
            sample_weight_dtype = X.dtype
        sample_weight = check_sample_weight(
            sample_weight,
            dtype=sample_weight_dtype,
            convert_dtype=convert_dtype,
            mem_type=mem_type,
            order=order,
        )
        out.append(sample_weight)

    check_consistent_length(*out)

    if return_classes:
        out.append(classes)
    if return_index:
        out.append(index)

    return out[0] if len(out) == 1 else tuple(out)
