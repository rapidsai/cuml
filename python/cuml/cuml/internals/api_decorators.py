#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import contextlib
import functools
import inspect
import typing

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
from cuml.internals.type_utils import _DecoratorType


def _wrap_once(wrapped, *args, **kwargs):
    """Prevent wrapping functions multiple times."""
    setattr(wrapped, CUML_WRAPPED_FLAG, True)
    return functools.wraps(wrapped, *args, **kwargs)


def _has_self(sig):
    return "self" in sig.parameters and list(sig.parameters)[0] == "self"


def _find_arg(sig, arg_name, default_position):
    params = list(sig.parameters)

    # Check for default name in input args
    if arg_name in sig.parameters:
        param = sig.parameters[arg_name]
        return arg_name, params.index(arg_name), param.default
    # Otherwise use argument in list by position
    elif arg_name is ...:
        index = int(_has_self(sig)) + default_position
        param = params[index]
        return param, index, sig.parameters[param].default
    else:
        raise ValueError(f"Unable to find parameter '{arg_name}'.")


def _get_value(args, kwargs, name, index, default_value, accept_lists=False):
    """Determine value for a given set of args, kwargs, name and index."""
    try:
        value = kwargs[name]
    except KeyError:
        try:
            value = args[index]
        except IndexError:
            if default_value is not inspect._empty:
                value = default_value
            else:
                raise IndexError(
                    f"Specified arg idx: {index}, and argument name: {name}, "
                    "were not found in args or kwargs."
                )
    # Accept list/tuple inputs when requested
    if accept_lists and isinstance(value, (list, tuple)):
        return np.asarray(value)

    return value


def _make_decorator(
    process_return=True,
    **defaults,
) -> typing.Callable[..., _DecoratorType]:
    # This function generates a function to be applied as decorator to a
    # wrapped function. For example:
    #
    #       a_decorator = _make_decorator(...)
    #
    #       ...
    #
    #       @a_decorator(...)  # apply decorator where appropriate
    #       def fit(X, y):
    #           ...
    #
    # Note: The decorator function can be partially closed by directly
    # providing keyword arguments to this function to be used as defaults.

    def decorator_function(
        input_arg: str = ...,
        get_output_type: bool = False,
        is_fit: bool = False,
    ) -> _DecoratorType:
        def decorator_closure(func):
            # This function constitutes the closed decorator that will return
            # the wrapped function. It performs function introspection at
            # function definition time. The code within the wrapper function is
            # executed at function execution time.

            # Prepare arguments
            sig = inspect.signature(func, follow_wrapped=True)

            has_self = _has_self(sig)

            if input_arg is not None and (is_fit or get_output_type):
                input_arg_ = _find_arg(sig, input_arg or "X", 0)
            else:
                input_arg_ = None

            @_wrap_once(func)
            def wrapper(*args, **kwargs):
                # Wraps the decorated function, executed at runtime.

                # Accept list/tuple inputs when accelerator is active
                accept_lists = cuml.accel.enabled()

                self_val = args[0] if has_self else None
                with InternalAPIContextBase(
                    func,
                    args,
                    is_base_method=isinstance(self_val, cuml.Base),
                    process_return=process_return,
                ) as cm:
                    if input_arg_:
                        input_val = _get_value(
                            args,
                            kwargs,
                            *input_arg_,
                            accept_lists=accept_lists,
                        )
                    else:
                        input_val = None

                    if is_fit:
                        assert self_val is not None
                        self_val._set_output_type(input_val)
                        self_val._set_n_features_in(input_val)

                    if get_output_type:
                        if self_val is None:
                            assert input_val is not None
                            out_type = iu.determine_array_type(input_val)
                        else:
                            out_type = self_val._get_output_type(input_val)

                        set_api_output_type(out_type)

                    if process_return:
                        ret = func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                return cm.process_return(ret)

            return wrapper

        return decorator_closure

    return functools.partial(decorator_function, **defaults)


# TODO:
# - infer get_output_type from whether return value is Base
# - determine why api_return_any is needed? It should only mark internal API?
# - infer `is_fit` based on method name by default
api_return_array = _make_decorator()
api_return_any = _make_decorator(process_return=False)
api_base_return_any = _make_decorator(is_fit=True, process_return=False)
api_base_return_array = _make_decorator(get_output_type=True)
api_base_fit_transform = _make_decorator(is_fit=True, get_output_type=True)
# TODO: investigate and remove these
api_base_return_any_skipall = api_return_any()
api_base_return_array_skipall = api_return_array()


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
