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
from cuml.internals.api_context_managers import (
    InternalAPIContextBase,
    set_api_output_type,
)
from cuml.internals.constants import CUML_WRAPPED_FLAG
from cuml.internals.global_settings import GlobalSettings

default = type(
    "default",
    (),
    dict.fromkeys(["__repr__", "__reduce__"], lambda s: "default"),
)()

VALID_OUTPUT_TYPES = (
    "array",
    "numba",
    "dataframe",
    "series",
    "df_obj",
    "cupy",
    "numpy",
    "cudf",
    "pandas",
)

INTERNAL_VALID_OUTPUT_TYPES = ("input", *VALID_OUTPUT_TYPES)


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
    if isinstance(output_type, str):
        output_type = output_type.lower()

    # Check for allowed types. Allow 'cuml' to support internal estimators
    if (
        output_type is not None
        and output_type != "cuml"
        and output_type not in INTERNAL_VALID_OUTPUT_TYPES
    ):
        valid_output_types_str = ", ".join(
            [f"'{x}'" for x in VALID_OUTPUT_TYPES]
        )
        raise ValueError(
            f"output_type must be one of {valid_output_types_str}"
            f" or None. Got: {output_type}"
        )

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


def _get_param(sig, name_or_index):
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


def reflect(
    func=None,
    *,
    array=default,
    model=default,
    reset=False,
    skip=False,
    default_output_type=None,
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
    default_output_type : str or None, default=None
        The default output type to use for a method when no output type
        has been set externally.
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
            default_output_type=default_output_type,
        )

    # TODO: remove this once auto-decorating is ripped out
    setattr(func, CUML_WRAPPED_FLAG, True)

    sig = inspect.signature(func, follow_wrapped=True)
    has_self = "self" in sig.parameters

    if model is default:
        model = "self" if has_self else None
    if model is not None:
        model = _get_param(sig, model)

    if array is default:
        if model is not None and list(sig.parameters).index(model) == 0:
            array = 1
        else:
            array = 0
        if len(sig.parameters) <= array:
            # Not enough parameters, no array-like param to infer from
            array = None
    if array is not None:
        array = _get_param(sig, array)

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

        if reset and model_arg is None:
            raise ValueError("`reset=True` is only valid on estimator methods")

        with InternalAPIContextBase(
            base=model_arg, process_return=not skip
        ) as cm:
            if reset:
                model_arg._set_output_type(array_arg)
                model_arg._set_n_features_in(array_arg)

            if model is not None:
                if array is not None:
                    out_type = model_arg._get_output_type(array_arg)
                else:
                    out_type = model_arg._get_output_type()
            elif array is not None:
                out_type = iu.determine_array_type(array_arg)
            elif default_output_type is not None:
                out_type = default_output_type
            else:
                out_type = None

            if out_type is not None:
                set_api_output_type(out_type)

            res = func(*args, **kwargs)

        if skip:
            return res
        return cm.process_return(res)

    return inner


def api_return_array(input_arg=default, get_output_type=False):
    return reflect(array=None if not get_output_type else input_arg)


def api_return_any():
    return reflect(array=None, skip=True)


def api_base_return_any():
    return reflect(reset=True)


def api_base_return_array(input_arg=default):
    return reflect(array="self" if input_arg is None else input_arg)


def api_base_fit_transform():
    return reflect(reset=True)


# TODO: investigate and remove these
api_base_return_any_skipall = api_return_any()
api_base_return_array_skipall = reflect


@contextlib.contextmanager
def exit_internal_api():
    assert GlobalSettings().root_cm is not None

    try:
        old_root_cm = GlobalSettings().root_cm

        GlobalSettings().root_cm = None

        # Set the global output type to the previous value to pretend we never
        # entered the API
        with using_output_type(old_root_cm.prev_output_type):
            yield

    finally:
        GlobalSettings().root_cm = old_root_cm
