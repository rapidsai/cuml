#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import contextlib
import functools
import inspect

import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cupy.cuda import Stream

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.global_settings import GlobalSettings

__all__ = (
    "check_output_type",
    "set_global_output_type",
    "using_output_type",
    "mlfunc",
    "ReflectedAttr",
    "reflect",
    "run_in_internal_context",
    "exit_internal_context",
    "enter_internal_context",
    "in_internal_context",
)


OUTPUT_TYPES = (
    "input",
    "numpy",
    "cupy",
    "cudf",
    "pandas",
    "numba",
    "array",
    "dataframe",
    "series",
    "df_obj",
)


def check_output_type(output_type: str) -> str:
    """Validate and normalize an ``output_type`` value"""
    # normalize as lower, keeping original str reference to appease the sklearn
    # standard estimator checks as much as possible.
    if output_type != (temp := output_type.lower()):
        output_type = temp
    # Check for allowed types. Allow 'cuml' to support internal estimators
    if output_type != "cuml" and output_type not in OUTPUT_TYPES:
        valid_output_types = ", ".join(map(repr, OUTPUT_TYPES))
        raise ValueError(
            f"`output_type` must be one of {valid_output_types}"
            f" or None. Got: {output_type!r}"
        )
    return output_type


def set_global_output_type(output_type):
    """Set the global output type.

    This output type will be used by functions and estimator methods.

    Note that instead of setting globally, an output type may be set
    contextually using :func:`using_output_type`, or on the estimator itself
    with the ``output_type`` parameter.

    Parameters
    ----------
    output_type : {'input', 'cupy', 'numpy', 'cudf', 'pandas', None}
        Desired output type of results and attributes of the estimators.

        * ``None``: No globally configured output type. This is the same as
          ``'input'``, except in cases where an estimator explicitly sets
          an ``output_type``.

        * ``'input'``: returns arrays of the same type as the inputs to the
          function or method. Fitted attributes will be of the same array type
          as ``X``.

        * ``'cupy'``: returns ``cupy`` arrays.

        * ``'numpy'``: returns ``numpy`` arrays.

        * ``'cudf'``: returns ``cudf.Series`` for single dimensional results
          and ``cudf.DataFrame`` otherwise.

        * ``'pandas'``: returns ``pandas.Series`` for single dimensional results
          and ``pandas.DataFrame`` otherwise.

    See Also
    --------
    cuml.using_output_type

    Notes
    -----
    ``cupy`` is the most efficient output type, as it supports flexible memory
    layouts and doesn't require device <-> host transfers.

    ``cudf`` has slightly more overhead for single dimensional outputs. For two
    dimensional outputs additional copies may be needed due to memory layout
    requirements of ``cudf.DataFrame``.

    ``numpy`` and ``pandas`` have a more significant overhead as they require
    device <-> host transfers. Whether that overhead matters is of course
    application specific.

    Examples
    --------
    >>> import cuml
    >>> import cupy as cp
    >>> import cudf
    >>> original_output_type = cuml.global_settings.output_type

    Fit a model with a cupy array. By default the fitted attributes will be
    cupy arrays.

    >>> X = cp.array([[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]])
    >>> model = cuml.DBSCAN(eps=1.0, min_samples=1).fit(X)
    >>> isinstance(model.labels_, cp.ndarray)
    True

    With a global output type set though, the fitted attributes will match
    the configured output type.

    >>> cuml.set_global_output_type("cudf")
    >>> isinstance(model.labels_, cudf.Series)
    True

    Reset the output type back to its original value.

    >>> cuml.set_global_output_type(original_output_type)
    """
    if output_type is not None:
        output_type = check_output_type(output_type)
    GlobalSettings().output_type = output_type


class using_output_type:
    """Configure the output type within a context.

    Parameters
    ----------
    output_type : {'input', 'cupy', 'numpy', 'cudf', 'pandas', None}
        Desired output type of results and attributes of the estimators.

        * ``None``: No globally configured output type. This is the same as
          ``'input'``, except in cases where an estimator explicitly sets
          an ``output_type``.

        * ``'input'``: returns arrays of the same type as the inputs to the
          function or method. Fitted attributes will be of the same array type
          as ``X``.

        * ``'cupy'``: returns ``cupy`` arrays.

        * ``'numpy'``: returns ``numpy`` arrays.

        * ``'cudf'``: returns ``cudf.Series`` for single dimensional results
          and ``cudf.DataFrame`` otherwise.

        * ``'pandas'``: returns ``pandas.Series`` for single dimensional results
          and ``pandas.DataFrame`` otherwise.

    See Also
    --------
    cuml.set_global_output_type

    Examples
    --------
    >>> import cuml
    >>> import cupy as cp
    >>> import cudf

    Fit a model with a cupy array. By default the fitted attributes will be
    cupy arrays.

    >>> X = cp.array([[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]])
    >>> model = cuml.DBSCAN(eps=1.0, min_samples=1).fit(X)
    >>> isinstance(model.labels_, cp.ndarray)
    True

    With a global output type set though, the fitted attributes will match
    the configured output type.

    >>> with cuml.using_output_type("cudf"):
    ...     print(isinstance(model.labels_, cudf.Series))
    True
    """

    def __init__(self, output_type):
        self.output_type = output_type

    def __enter__(self):
        self.prev_output_type = GlobalSettings().output_type
        set_global_output_type(self.output_type)
        return self.prev_output_type

    def __exit__(self, *_):
        GlobalSettings().output_type = self.prev_output_type


@contextlib.contextmanager
def enter_internal_context():
    """Enter an internal context.

    Returns ``True`` if this is a new internal context, or ``False``
    if the code was already running within an internal context."""
    gs = GlobalSettings()
    if gs._external_output_type is False:
        # External, this is a new context
        gs._external_output_type = gs.output_type
        gs.output_type = "mirror"
        try:
            with Stream.ptds:
                yield True
        finally:
            gs.output_type = gs._external_output_type
            gs._external_output_type = False
    else:
        # Already internal, just yield
        yield False


def in_internal_context() -> bool:
    """Returns True if running in an internal context."""
    return GlobalSettings()._external_output_type is not False


@contextlib.contextmanager
def exit_internal_context():
    """Exit an internal context.

    Code run in this context will run under the original
    configuration before an internal context was entered"""
    gs = GlobalSettings()
    if gs._external_output_type is False:
        # Already external, nothing to do
        yield
    else:
        orig_external_output_type = gs._external_output_type
        orig_output_type = gs.output_type
        gs.output_type = orig_external_output_type
        gs._external_output_type = False
        try:
            yield
        finally:
            gs._external_output_type = orig_external_output_type
            gs.output_type = orig_output_type


def infer_output_type(array, array_like="numpy"):
    """Infer the corresponding ``output_type`` given an input array-like.

    Parameters
    ----------
    array : array-like
        The array-like value to infer from.
    array_like : Any, default="numpy"
        The value to return if `array` is not an array but is array-like.

    Returns
    -------
    output_type : {"cupy", "numpy", "pandas", "cudf", "numba", "cuml", None}
        The inferred ``output_type``, or ``None`` if not an array-like input.
    """
    if isinstance(array, np.ndarray) or sp.issparse(array):
        return "numpy"
    elif isinstance(array, cp.ndarray) or cp_sp.issparse(array):
        return "cupy"
    elif isinstance(array, (CumlArray, SparseCumlArray)):
        return "cuml"
    elif isinstance(array, (cudf.DataFrame, cudf.Series, cudf.Index)):
        return "cudf"
    elif isinstance(array, (pd.DataFrame, pd.Series, pd.Index)):
        return "pandas"
    elif hasattr(array, "__cuda_ndarray__"):
        return "numba"
    elif hasattr(array, "__cuda_array_interface__"):
        return "cupy"

    # Explicitly exclude a few common collections that aren't array-likes. This
    # matches those also explicitly excluded in our validation routines.
    if isinstance(array, (str, bytes, dict)):
        return None

    # Exclude numpy scalars, which also implement `__array__`
    if np.isscalar(array):
        return None

    # Types with any of these attributes _may_ be coerced to an array by our
    # validation methods (e.g. `check_array`). The actual instance may error at
    # that point, but that's fine, this is just a best effort inference to
    # exclude non-array-like things like `None`/`1`/...
    for name in ["__array__", "__array_interface__", "__len__"]:
        if hasattr(array, name):
            return array_like

    # Not an array-like input, just return None
    return None


class ArrayIndexPair:
    """An array paired with an aligned index.

    Used to attach `index` metadata to an `array` for use when returning
    dataframe-like outputs.

    Parameters
    ----------
    array : cupy.ndarray
        An array input.
    index : pandas.Index, cudf.Index, or None
        An index to attach. Must have the same length as `array`.
    """

    def __init__(self, array, index):
        self.array = array
        self.index = index


class ClassLabels:
    """An output type for reflecting class labels.

    This is a workaround for a current limitation in `cupy` - it cannot handle
    non-numeric dtypes. As such, methods that return class labels (which support
    non-numeric dtypes) need to return something other than a `cupy.ndarray`.

    This type plugs in to the output conversion machinery, delaying conversion
    to a specific `output_type` until necessary. If a type doesn't support
    the requested `dtype`, an appropriate error is raised.

    Parameters
    ----------
    indices : cupy.ndarray
        An array of labels encoded as indices into `classes`.
    classes : numpy.ndarray
        An array of classes.
    """

    def __init__(self, indices, classes):
        self.indices = indices
        self.classes = classes

    def to_output(self, output_type=None, index=None):
        """Convert this instance to a specific `output_type`.

        Parameters
        ----------
        output_type : {'cupy', 'numpy', 'cudf', 'pandas', 'numba'} or None
            The output type to convert to. If `None`, `cupy` will be used when
            possible, falling back to `cudf` if necessary.
        index : pandas.Index, cudf.Index, or None, default=None
            An optional index to attach to arrays when returning dataframe-like
            outputs.

        Returns
        -------
        labels
            The class labels stored in the requested output type.
        """
        if output_type == "cuml":
            return self

        if isinstance(self.classes, list):
            # Multi-target output
            dtype = (
                self.classes[0].dtype
                if len(set(c.dtype for c in self.classes)) == 1
                else None
            )
            if dtype is not None and dtype.kind in "iufb":
                # All dtypes are identical and numeric, we can use cupy here
                if all((c == np.arange(len(c))).all() for c in self.classes):
                    # Fast path for common case of monotonically increasing numeric classes
                    labels = self.indices.astype(dtype, copy=False)
                else:
                    # Need to transform indices back to classes
                    labels = cp.empty(shape=self.indices.shape, dtype=dtype)
                    for i, c in enumerate(self.classes):
                        labels[:, i] = cp.asarray(c).take(self.indices[:, i])

                out = labels
            else:
                # At least one class is non-numeric, we need to use cudf
                out = cudf.DataFrame(
                    {
                        i: cudf.Series(
                            c,
                            dtype=("object" if c.dtype.kind in "OU" else None),
                        )
                        .take(self.indices[:, i])
                        .reset_index(drop=True)
                        for i, c in enumerate(self.classes)
                    },
                    index=index,
                )
        else:
            # Single-target output
            dtype = self.classes.dtype
            if self.classes.dtype.kind in "iufb":
                # Numeric dtype, we can use cupy here
                if (self.classes == np.arange(len(self.classes))).all():
                    # Fast path for common case of monotonically increasing numeric classes
                    labels = self.indices.astype(
                        self.classes.dtype, copy=False
                    )
                else:
                    # Need to transform indices back to classes
                    labels = cp.asarray(self.classes).take(self.indices)

                out = labels
            else:
                # Non-numeric classes. We use cudf since it supports all types, and will
                # error appropriately later on when converting to outputs like `cupy`
                # that don't support strings.
                cudf_dtype = (
                    "object" if self.classes.dtype.kind in "OU" else None
                )
                out = (
                    cudf.Series(self.classes, dtype=cudf_dtype)
                    .take(self.indices)
                    .reset_index(drop=True)
                )
                if index is not None:
                    out.index = index

        if output_type is None:
            # cupy when possible, cudf otherwise
            return out

        # Coerce result to requested output_type
        if isinstance(out, cp.ndarray):
            return convert_arrays(out, output_type, index=index)
        elif output_type in ("cudf", "df_obj"):
            return out
        elif output_type == "dataframe":
            return out.to_frame() if isinstance(out, cudf.Series) else out
        elif output_type == "series" and isinstance(out, cudf.Series):
            return out
        elif output_type == "pandas":
            if cudf.pandas.LOADED:
                return cudf.pandas.as_proxy_object(out)
            return out.to_pandas()
        elif output_type in ("numpy", "array"):
            # XXX: dtype coercion not needed for object, and when specified
            # cudf will sometimes coerce `None -> <NA>` erroneously.
            # See https://github.com/rapidsai/cudf/issues/22419
            # Better to leave unspecified in this case.
            return out.to_numpy(dtype=None if dtype == "object" else dtype)
        else:
            raise TypeError(
                f"{output_type=!r} doesn't support outputs of dtype "
                f"{dtype or 'object'} and shape {self.indices.shape}"
            )


def convert_arrays(obj, output_type="cupy", index=None, _legacy=False):
    """Convert arrays in `obj` to the specified `output_type`.

    Parameters
    ----------
    obj : object
        The object to convert. Any cupy arrays (dense or sparse) or
        cuml-specific output types (`ClassLabels`, `ArrayIndexPair`) will be
        converted to the specified `output_type`. Some builtin collections
        (dict, list, tuple) are traversed recursively to find array-likes.
        Other array-likes (numpy, pandas, ...) will error as unsupported.
        Any other type is passed through unchanged.
    output_type : {'cupy', 'numpy', 'cudf', 'pandas', 'numba'}
        The output type to convert to.
    index : pandas.Index, cudf.Index, or None, default=None
        An optional index to attach to arrays when returning dataframe-like
        outputs.

    Returns
    -------
    out : object
        The equivalent output, with arrays coerced to `output_type`.
    """
    # TODO: legacy output paths, remove once `CumlArray`/`SparseCumlArray`
    # are no longer used anywhere.
    if isinstance(obj, CumlArray):
        assert index is None
        return obj.to_output(output_type)
    elif isinstance(obj, SparseCumlArray):
        return obj.to_output(
            "scipy" if output_type in ["numpy", "pandas"] else "cupy"
        )

    if isinstance(obj, ArrayIndexPair):
        index = obj.index
        obj = obj.array

    if isinstance(obj, ClassLabels):
        return obj.to_output(output_type, index=index)

    elif isinstance(obj, cp.ndarray):
        if output_type == "numpy":
            return obj.get(order="A")
        elif output_type in (
            "cudf",
            "pandas",
            "df_obj",
            "dataframe",
            "series",
        ):
            if output_type == "series":
                if obj.ndim == 2:
                    if obj.shape[1] == 1:
                        obj = obj.flatten()
                    else:
                        raise ValueError(
                            "Only single dimensional arrays can be transformed to"
                            " Series."
                        )
                elif obj.ndim == 0:
                    obj = obj[None]
            elif output_type == "dataframe":
                if obj.ndim == 1:
                    obj = obj[:, None]
                elif obj.ndim == 0:
                    obj = obj[None, None]

            if obj.ndim == 2:
                if obj.shape[1] == 1 and output_type != "dataframe":
                    df = cudf.Series(obj.flatten(), index=index)
                else:
                    df = cudf.DataFrame(obj, index=index)
            else:
                df = cudf.Series(obj, index=index)

            if output_type == "pandas":
                if cudf.pandas.LOADED:
                    return cudf.pandas.as_proxy_object(df)
                return df.to_pandas()
            return df

        elif output_type == "numba":
            from numba import cuda

            return cuda.as_cuda_array(obj)
        elif output_type == "cuml" and _legacy:
            # TODO: remove legacy output path once all consumers are updated
            return CumlArray.from_input(obj, order="K")
        else:
            assert output_type in ("cuml", "cupy", "array")
            # Return `cupy` directly
            return obj

    elif cp_sp.issparse(obj):
        if output_type in ("numpy", "pandas"):
            return obj.get()
        elif output_type == "cuml" and _legacy:
            # TODO: remove legacy output path once all consumers are updated
            return SparseCumlArray(obj)
        else:
            return obj

    elif isinstance(
        obj,
        (np.ndarray, cudf.Series, cudf.DataFrame, pd.Series, pd.DataFrame),
    ):
        raise TypeError(
            f"Cannot return objects of type {type(obj).__name__} directly "
            f"from an `mlfunc`-decorated function. Please return a "
            f"`cupy.ndarray`, `cupyx.scipy.sparse.spmatrix`, `ArrayIndexPair`, "
            f"or `ClassLabels` instead."
        )

    elif isinstance(obj, list):
        return [convert_arrays(v, output_type, index, _legacy) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(
            convert_arrays(v, output_type, index, _legacy) for v in obj
        )

    elif isinstance(obj, dict):
        return {
            k: convert_arrays(v, output_type, index, _legacy)
            for k, v in obj.items()
        }

    else:
        return obj


class ReflectedAttr:
    """A descriptor for enabling `output_type` reflection on an attribute."""

    class Cache:
        """A cache for conversions of values per `output_type`"""

        def __init__(self, value):
            self.value = value
            self._cache = {} if self._requires_reflection(value) else None

        def _requires_reflection(self, obj) -> bool:
            """Check if `obj` requires reflection."""
            if isinstance(obj, (cp.ndarray, ArrayIndexPair)):
                return True
            elif cp_sp.issparse(obj):
                return True
            elif isinstance(
                obj,
                (
                    np.ndarray,
                    cudf.Series,
                    cudf.DataFrame,
                    pd.Series,
                    pd.DataFrame,
                ),
            ):
                raise TypeError(
                    "Array-like types other than cupy, cupyx.scipy.sparse, or "
                    "`ArrayIndexPair` are not supported."
                )
            elif isinstance(obj, (list, tuple)):
                return any(self._requires_reflection(v) for v in obj)
            elif isinstance(obj, dict):
                return any(self._requires_reflection(v) for v in obj.values())
            return False

        def get(self, output_type):
            """Get the proper value for a given `output_type` from the cache"""
            if self._cache is None:
                # No need to reflect
                return self.value
            if output_type in self._cache:
                return self._cache[output_type]
            out = convert_arrays(self.value, output_type)
            if not (isinstance(out, (list, tuple, set)) or cp.isscalar(out)):
                self._cache[output_type] = out
            return out

        def __reduce__(self):
            return (ReflectedAttr.Cache, (self.value,))

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self

        cache = instance.__dict__.get(self.name)
        if cache is None:
            raise AttributeError(
                f"{owner.__name__!r} object has no attribute {self.name!r}"
            )

        output_type = GlobalSettings().output_type

        if output_type == "mirror":
            return cache.value
        if output_type is None:
            output_type = instance.output_type
        if output_type in (None, "input"):
            output_type = instance._input_type

        return cache.get(output_type)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = self.Cache(value)

    def __delete__(self, instance):
        if instance.__dict__.pop(self.name, None) is None:
            raise AttributeError(
                f"{type(instance).__name__!r} object has no attribute {self.name!r}"
            )


def _get_param(sig, name_or_index):
    """Get an `inspect.Parameter` instance by name or index from a
    signature, and validates it's not variadic.

    Used for normalizing `array`/`model` args to `reflect`."""
    if isinstance(name_or_index, str):
        param = sig.parameters[name_or_index]
    else:
        param = list(sig.parameters.values())[name_or_index]

    if param.kind in (
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        raise ValueError("Cannot reflect variadic args/kwargs")

    return param.name


def mlfunc(
    func=None,
    *,
    model_arg=...,
    array_arg=...,
    convert_output=True,
    set_input_type=False,
    preserve_index=False,
    _legacy=False,
):
    """A decorator for enabling common `cuml` machinery on a function/method.

    This decorator should be applied to any public function or method that
    invokes a CUDA kernel (through either `libcuml`, `cupy`, or `cudf`),
    or accesses any functions or attributes that require type reflection.
    In short - most public functions or methods.

    Functions and methods decorated with this get a few additional behaviors:

    - They are run within an "internal context". This mainly means that
      reflected functions/methods or estimator fitted attributes will be
      returned consistently as `cupy` instances instead of their reflected
      types.

    - Their output is converted to the configured/inferred output type. If not
      needed, conversion may be disabled by setting ``convert_output=False``.

    The decorator also eases compliance with a few standard ``cuml`` API
    conventions:

    - Fit-like methods on estimators should store the required metadata like
      ``_input_type`` to support cases like ``output_type="input"``. To enable
      this for a method set ``set_input_type=True``.

    - Inference methods should preserve the index of the input in the output.
      To enable this for a method, set ``preserve_index=True``.

    Parameters
    ----------
    func : callable or None
        The function to be decorated, or None to curry to be applied later.
    model_arg : int, str, or None, default=...
        The ``cuml.Base`` parameter to infer the reflected output type from. By
        default this will be ``'self'`` (if present), and ``None`` otherwise.
        Provide a parameter position or name to override. May also provide
        ``None`` to disable this inference entirely.
    array_arg : int, str, or None, default=...
        The array-like parameter to infer the reflected output type from. By
        default this will be the first argument to the method or function
        (excluding ``'self'`` or ``model_arg``), or ``None`` if there are no other
        arguments. Provide a parameter position or name to override. May also
        provide ``None`` to disable this inference entirely; in this case the
        output type is expected to be specified manually either internal or
        external to the method.
    convert_output : bool, default=True
        Whether to convert the function output to the configured/inferred
        output type.
    set_input_type : bool, default=False
        If True, the input type for reflection is reset on the estimator.
        Defaults to False, to not reset anything. Most estimators should set
        ``set_input_type=True`` on any fit-like methods.
    preserve_index : bool, default=False
        Whether to preserve the index of the ``array_arg`` argument (if any)
        in the function output. This should typically be set to ``True`` on
        any inference (predict/transform-like) methods.
    """
    if func is None:
        return lambda func: mlfunc(
            func,
            model_arg=model_arg,
            array_arg=array_arg,
            convert_output=convert_output,
            preserve_index=preserve_index,
            set_input_type=set_input_type,
            _legacy=_legacy,
        )

    sig = inspect.signature(func, follow_wrapped=True)

    # Normalize model_arg to str | None
    if model_arg is ...:
        model_arg = "self" if ("self" in sig.parameters) else None
    if model_arg is not None:
        model_arg = _get_param(sig, model_arg)

    # Normalize array_arg to str | None
    if array_arg is ...:
        array_arg = int(
            model_arg is not None
            and list(sig.parameters).index(model_arg) == 0
        )
        if len(sig.parameters) <= array_arg:
            # Not enough parameters, no array-like param to infer from
            array_arg = None
    if array_arg is not None:
        array_arg = _get_param(sig, array_arg)

    if set_input_type and (model_arg is None or array_arg is None):
        raise ValueError(
            "`set_input_type=True` is not valid with `array_arg=None` "
            "or `model_arg=None`"
        )

    if preserve_index and array_arg is None:
        raise ValueError(
            "`preserve_index=True` is not valid with `array_arg=None`"
        )

    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Bind arguments
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        model = None if model_arg is None else bound.arguments[model_arg]
        array = None if array_arg is None else bound.arguments[array_arg]

        if preserve_index and isinstance(
            array, (cudf.Series, cudf.DataFrame, pd.Series, pd.DataFrame)
        ):
            index = array.index
        else:
            index = None

        with enter_internal_context() as was_external:
            if set_input_type is True:
                model._set_output_type(array)

            res = func(*args, **kwargs)

        if convert_output:
            gs = GlobalSettings()
            if was_external or gs.output_type != "mirror":
                # We're returning to the user, infer the expected output type
                if model_arg is not None:
                    if array_arg is not None:
                        output_type = model._get_output_type(array)
                    else:
                        output_type = model._get_output_type()
                else:
                    output_type = gs.output_type
                    if output_type in ("input", None):
                        if array_arg is not None:
                            output_type = infer_output_type(array)
                        if output_type in ("input", None):
                            # Nothing to infer from and no explicit type set,
                            # default to cupy
                            output_type = "cupy"
            else:
                # We're internal
                output_type = "cuml"

            with enter_internal_context():
                res = convert_arrays(
                    res, output_type, index=index, _legacy=_legacy
                )

        return res

    return inner


def run_in_internal_context(func):
    """Decorate a function to run within an "internal context".

    New code should use `mlfunc` instead, this decorator will go away in the
    near future.

    An "internal context" mainly means that reflected functions/methods or
    estimator fitted attributes will be returned as ``CumlArray`` instances
    instead of their reflected types.

    Unlike `reflect`, functions decorated with this do not participate in the
    reflection system.
    """
    return mlfunc(func, convert_output=False)


def reflect(
    func=None,
    *,
    array=...,
    model=...,
    reset=False,
):
    """Mark a function or method as participating in the reflection system.

    New code should use `mlfunc` instead, this decorator will go away in the
    near future.

    Functions and methods decorated with this get a few additional behaviors:

    - They are run within an "internal context". This mainly means that
      reflected functions/methods or estimator fitted attributes will be
      returned as ``CumlArray`` instances instead of their reflected types. If
      this is the only behavior you want, you should use
      `run_in_internal_context` instead.

    - Their output type is converted to the proper output type following
      standard cuml behavior. The default behavior covers most cases, but when
      needed you may want to specify the ``model`` and/or ``array`` parameters
      manually.

    - For estimators, fit-like methods will store the required metadata like
      ``_input_type`` to support cases like ``output_type="input"``. To enable
      this for a method set ``reset=True``.

    Parameters
    ----------
    func : callable or None
        The function to be decorated, or None to curry to be applied later.
    model : int, str, or None, default=...
        The ``cuml.Base`` parameter to infer the reflected output type from. By
        default this will be ``'self'`` (if present), and ``None`` otherwise.
        Provide a parameter position or name to override. May also provide
        ``None`` to disable this inference entirely.
    array : int, str, or None, default=...
        The array-like parameter to infer the reflected output type from. By
        default this will be the first argument to the method or function
        (excluding ``'self'`` or ``model``), or ``None`` if there are no other
        arguments. Provide a parameter position or name to override. May also
        provide ``None`` to disable this inference entirely; in this case the
        output type is expected to be specified manually either internal or
        external to the method.
    reset : bool, default=False
        If True, the input type for reflection is reset on the estimator.
        Defaults to False, to not reset anything. Most estimators should set
        ``reset=True`` on any fit-like methods.
    """
    return mlfunc(
        func,
        model_arg=model,
        array_arg=array,
        set_input_type=reset,
        _legacy=True,
    )
