#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import contextlib
import functools
import inspect

import numpy as np

import cuml
import cuml.accel

# TODO: Try to resolve circular import that makes this necessary:
from cuml.internals import input_utils as iu
from cuml.internals.api_context_managers import (
    InternalAPIContextBase,
    set_api_output_type,
)
from cuml.internals.constants import CUML_WRAPPED_FLAG
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.memory_utils import using_output_type

default = type(
    "default",
    (),
    dict.fromkeys(["__repr__", "__reduce__"], lambda s: "default"),
)()


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
