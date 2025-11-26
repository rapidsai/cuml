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


default = type(
    "default",
    (),
    dict.fromkeys(["__repr__", "__reduce__"], lambda s: "default"),
)()

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
    # normalize as lower, keeping original str reference to appease the sklearn
    # standard estimator checks as much as possible.
    if output_type != (temp := output_type.lower()):
        output_type = temp
    # Check for allowed types. Allow 'cuml' to support internal estimators
    if output_type != "cuml" and output_type not in OUTPUT_TYPES:
        valid_output_types = ", ".join(map(repr, OUTPUT_TYPES))
        raise ValueError(
            f"output_type must be one of {valid_output_types}"
            f" or None. Got: {output_type}"
        )
    return output_type


def set_global_output_type(output_type):
    """
    Method to set cuML's single GPU estimators global output type.
    It will be used by all estimators unless overridden in their initialization
    with their own output_type parameter. Can also be overridden by the context
    manager method :func:`using_output_type`.

    Parameters
    ----------
    output_type : {'input', 'cudf', 'cupy', 'numpy'} (default = 'input')
        Desired output type of results and attributes of the estimators.

        * ``'input'`` will mean that the parameters and methods will mirror the
          format of the data sent to the estimators/methods as much as
          possible. Specifically:

          +---------------------------------------+--------------------------+
          | Input type                            | Output type              |
          +=======================================+==========================+
          | cuDF DataFrame or Series              | cuDF DataFrame or Series |
          +---------------------------------------+--------------------------+
          | NumPy arrays                          | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Pandas DataFrame or Series            | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Numba device arrays                   | Numba device arrays      |
          +---------------------------------------+--------------------------+
          | CuPy arrays                           | CuPy arrays              |
          +---------------------------------------+--------------------------+
          | Other `__cuda_array_interface__` objs | CuPy arrays              |
          +---------------------------------------+--------------------------+

        * ``'cudf'`` will return cuDF Series for single dimensional results and
          DataFrames for the rest.

        * ``'cupy'`` will return CuPy arrays.

        * ``'numpy'`` will return NumPy arrays.

    Examples
    --------
    >>> import cuml
    >>> import cupy as cp
    >>> ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    >>> ary = cp.asarray(ary)
    >>> prev_output_type = cuml.global_settings.output_type
    >>> cuml.set_global_output_type('cudf')
    >>> dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    >>> dbscan_float.fit(ary)
    DBSCAN()
    >>>
    >>> # cuML output type
    >>> dbscan_float.labels_
    0    0
    1    1
    2    2
    dtype: int32
    >>> type(dbscan_float.labels_)
    <class 'cudf.core.series.Series'>
    >>> cuml.set_global_output_type(prev_output_type)

    Notes
    -----
    ``'cupy'`` and ``'numba'`` options (as well as ``'input'`` when using Numba
    and CuPy ndarrays for input) have the least overhead. cuDF add memory
    consumption and processing time needed to build the Series and DataFrames.
    ``'numpy'`` has the biggest overhead due to the need to transfer data to
    CPU memory.

    """
    if output_type is not None:
        output_type = check_output_type(output_type)
    GlobalSettings().output_type = output_type


class using_output_type:
    """
    Context manager method to set cuML's global output type inside a `with`
    statement. It gets reset to the prior value it had once the `with` code
    block is executer.

    Parameters
    ----------
    output_type : {'input', 'cudf', 'cupy', 'numpy'} (default = 'input')
        Desired output type of results and attributes of the estimators.

        * ``'input'`` will mean that the parameters and methods will mirror the
          format of the data sent to the estimators/methods as much as
          possible. Specifically:

          +---------------------------------------+--------------------------+
          | Input type                            | Output type              |
          +=======================================+==========================+
          | cuDF DataFrame or Series              | cuDF DataFrame or Series |
          +---------------------------------------+--------------------------+
          | NumPy arrays                          | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Pandas DataFrame or Series            | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Numba device arrays                   | Numba device arrays      |
          +---------------------------------------+--------------------------+
          | CuPy arrays                           | CuPy arrays              |
          +---------------------------------------+--------------------------+
          | Other `__cuda_array_interface__` objs | CuPy arrays              |
          +---------------------------------------+--------------------------+

        * ``'cudf'`` will return cuDF Series for single dimensional results and
          DataFrames for the rest.

        * ``'cupy'`` will return CuPy arrays.

        * ``'numpy'`` will return NumPy arrays.

    Examples
    --------
    >>> import cuml
    >>> import cupy as cp
    >>> ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    >>> ary = cp.asarray(ary)
    >>> with cuml.using_output_type('cudf'):
    ...     dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    ...     dbscan_float.fit(ary)
    ...
    ...     print("cuML output inside 'with' context")
    ...     print(dbscan_float.labels_)
    ...     print(type(dbscan_float.labels_))
    ...
    DBSCAN()
    cuML output inside 'with' context
    0    0
    1    1
    2    2
    dtype: int32
    <class 'cudf.core.series.Series'>
    >>> # use cuml again outside the context manager
    >>> dbscan_float2 = cuml.DBSCAN(eps=1.0, min_samples=1)
    >>> dbscan_float2.fit(ary)
    DBSCAN()
    >>> # cuML default output
    >>> dbscan_float2.labels_
    array([0, 1, 2], dtype=int32)
    >>> isinstance(dbscan_float2.labels_, cp.ndarray)
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

    return res.to_output(output_type=output_type)


def reflect(
    func=None,
    *,
    array=default,
    model=default,
    reset=False,
    skip=False,
):
    """Mark a function or method as participating in the reflection system.

    Parameters
    ----------
    func : callable or None
        The function to be decorated, or None to curry to be applied later.
    model : int, str, or None, default=default
        The ``cuml.Base`` parameter to infer the reflected output type from. By
        default this will be ``'self'`` (if present), and ``None`` otherwise.
        Provide a parameter position or name to override. May also provide
        ``None`` to disable this inference entirely.
    array : int, str, or None, default=default
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
    skip : bool, default=False
        Set to True to skip output processing for a method. This is mostly
        useful if output processing will be handled manually.
    """
    # Local to avoid circular imports
    import cuml.accel

    if func is None:
        return lambda func: reflect(
            func,
            model=model,
            array=array,
            reset=reset,
            skip=skip,
        )

    sig = inspect.signature(func, follow_wrapped=True)
    has_self = "self" in sig.parameters

    # Normalize model to str | None
    if model is default:
        if skip and not reset:
            # We're skipping output processing and not resetting an estimator,
            # there's no need to touch input parameters at all
            model = None
        else:
            model = "self" if has_self else None
    if model is not None:
        model = _get_param(sig, model)

    # Normalize array to str | None
    if array is default:
        if skip and not reset:
            array = None
        else:
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

        if not skip:
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

            res = coerce_arrays(res, output_type)

        return res

    return inner
