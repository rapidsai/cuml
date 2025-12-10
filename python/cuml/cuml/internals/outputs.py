#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import contextlib
import functools
import inspect

import numpy as np

# TODO: Try to resolve circular import that makes this necessary:
from cuml.internals import input_utils as iu
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.global_settings import GlobalSettings

__all__ = (
    "check_output_type",
    "set_global_output_type",
    "using_output_type",
    "reflect",
    "exit_internal_api",
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
def enter_internal_api():
    """Enter an internal API context.

    Returns ``True`` if this is a new internal context, or ``False``
    if the code was already running within an internal context."""
    gs = GlobalSettings()
    if gs._external_output_type is False:
        # External, this is a new context
        gs._external_output_type = gs.output_type
        gs.output_type = "mirror"
        try:
            yield True
        finally:
            gs.output_type = gs._external_output_type
            gs._external_output_type = False
    else:
        # Already internal, just yield
        yield False


@contextlib.contextmanager
def exit_internal_api():
    """Exit an internal API context.

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


def coerce_arrays(res, output_type):
    """Traverse a result, converting it to the proper output type"""
    if isinstance(res, tuple):
        return tuple(coerce_arrays(i, output_type) for i in res)
    elif isinstance(res, list):
        return [coerce_arrays(i, output_type) for i in res]
    elif isinstance(res, dict):
        return {k: coerce_arrays(v, output_type) for k, v in res.items()}

    # Get the output type
    arr_type, is_sparse = iu.determine_array_type_full(res)

    if arr_type is None:
        # Not an array, just return
        return res

    # If we are a supported array and not already cuml, convert to cuml
    if arr_type != "cuml":
        if is_sparse:
            res = SparseCumlArray(res, convert_index=False)
        else:
            res = iu.input_to_cuml_array(res, order="K").array

    if output_type == "cuml":
        # Return CumlArray/SparseCumlArray directly
        return res

    if is_sparse:
        # Coerce output_type to supported sparse types.
        # Host types -> scipy, cupy otherwise.
        output_type = "scipy" if output_type in ["numpy", "pandas"] else "cupy"

    return res.to_output(output_type=output_type)


def run_in_internal_api(func):
    """Decorate a function to run within an "internal API context".

    This mainly means that reflected functions/methods or estimator fitted
    attributes will be returned as ``CumlArray`` instances instead of their
    reflected types.

    Unlike `reflect`, functions decorated with this do not participate in the
    reflection system.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        with enter_internal_api():
            return func(*args, **kwargs)

    return inner


def reflect(
    func=None,
    *,
    array=...,
    model=...,
    reset=False,
):
    """Mark a function or method as participating in the reflection system.

    Functions and methods decorated with this get a few additional behaviors:

    - They are run within an "internal API context". This mainly means that
      reflected functions/methods or estimator fitted attributes will be
      returned as ``CumlArray`` instances instead of their reflected types. If
      this is the only behavior you want, you should use `run_in_internal_api`
      instead.

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
        Set to True for methods like ``fit`` that reset the reflected type on
        an estimator.
    """
    # Local to avoid circular imports
    import cuml.accel

    if func is None:
        return lambda func: reflect(
            func,
            model=model,
            array=array,
            reset=reset,
        )

    sig = inspect.signature(func, follow_wrapped=True)

    # Normalize model to str | None
    if model is ...:
        model = "self" if ("self" in sig.parameters) else None
    if model is not None:
        model = _get_param(sig, model)

    # Normalize array to str | None
    if array is ...:
        array = int(
            model is not None and list(sig.parameters).index(model) == 0
        )
        if len(sig.parameters) <= array:
            # Not enough parameters, no array-like param to infer from
            array = None
    if array is not None:
        array = _get_param(sig, array)

    if reset and (model is None or array is None):
        raise ValueError(
            "`reset=True` is not valid with `array=None` or `model=None`"
        )

    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Accept list/tuple inputs when accelerator is active
        accept_lists = cuml.accel.enabled()

        # Bind arguments
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        model_arg = None if model is None else bound.arguments[model]
        array_arg = None if array is None else bound.arguments[array]
        if accept_lists and isinstance(array_arg, (list, tuple)):
            array_arg = np.asarray(array_arg)

        with enter_internal_api() as was_external:
            if reset:
                model_arg._set_output_type(array_arg)
                model_arg._set_n_features_in(array_arg)

            res = func(*args, **kwargs)

        gs = GlobalSettings()
        if was_external or gs.output_type != "mirror":
            # We're returning to the user, infer the expected output type
            if model is not None:
                if array is not None:
                    output_type = model_arg._get_output_type(array_arg)
                else:
                    output_type = model_arg._get_output_type()
            else:
                output_type = gs.output_type
                if output_type in ("input", None):
                    if array is not None:
                        output_type = iu.determine_array_type(array_arg)
                    if output_type in ("input", None):
                        # Nothing to infer from and no explicit type set,
                        # default to cupy
                        output_type = "cupy"
        else:
            # We're internal, return as cuml
            output_type = "cuml"

        return coerce_arrays(res, output_type)

    return inner
