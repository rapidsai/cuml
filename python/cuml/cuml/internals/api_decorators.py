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


def reflect(func=None, *, on=0, reset=False, skip=False):
    """Mark a function or method as participating in the reflection system.

    Parameters
    ----------
    func : callable or None
        The function to be decorated, or None to curry to be applied later.
    on : int, str, or None, default=0
        The parameter to infer the reflected output type from. By default this
        will be the first argument to the method or function (excluding
        ``self``, unless there are no other arguments). Provide a parameter
        position or name to override. May also provide None to disable
        this inference entirely; in this case the output type is expected
        to be specified manually either internal or external to the method.
    reset : bool, default=False
        Set to True for methods like ``fit`` that reset the reflected type on
        an estimator.
    skip : bool, default=False
        Set to True to skip output processing for a method. This is mostly
        useful if output processing will be handled manually.
    """
    if func is None:
        return lambda func: reflect(func, on=on, reset=reset, skip=skip)

    # TODO: remove this once auto-decorating is ripped out
    setattr(func, CUML_WRAPPED_FLAG, True)

    sig = inspect.signature(func, follow_wrapped=True)

    if on is not None:
        has_self = "self" in sig.parameters

        if isinstance(on, str):
            param = sig.parameters[on]
        elif on == 0 and has_self and len(sig.parameters) == 1:
            # Default to self if there are no other parameters
            param = sig.parameters["self"]
        else:
            # Otherwise exclude self, defaulting to first parameter
            param = list(sig.parameters.values())[on + has_self]

        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise ValueError("Cannot reflect on variadic args/kwargs")

        on = param.name

    @functools.wraps(func)
    def inner(*args, **kwargs):
        # Accept list/tuple inputs when accelerator is active
        accept_lists = cuml.accel.enabled()

        # Bind arguments
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if on is None:
            base = None
        else:
            on_arg = bound.arguments[on]
            if accept_lists and isinstance(on_arg, (list, tuple)):
                on_arg = np.asarray(on_arg)

            # Look for an estimator, first in `on` and then in `self`
            if isinstance(on_arg, cuml.Base):
                base = on_arg
            elif has_self and isinstance(bound.arguments["self"], cuml.Base):
                base = bound.arguments["self"]
            else:
                base = None

        if reset and base is None:
            raise ValueError("`reset=True` is only valid on estimator methods")

        with InternalAPIContextBase(base=base, process_return=not skip) as cm:
            if reset:
                base._set_output_type(on_arg)
                base._set_n_features_in(on_arg)

            if on is not None:
                if isinstance(on_arg, cuml.Base):
                    out_type = on_arg._get_output_type()
                elif base is not None:
                    out_type = base._get_output_type(on_arg)
                else:
                    out_type = iu.determine_array_type(on_arg)

                set_api_output_type(out_type)

            res = func(*args, **kwargs)

        if skip:
            return res
        return cm.process_return(res)

    return inner


def api_return_array(input_arg=0, get_output_type=False):
    return reflect(on=None if not get_output_type else input_arg)


def api_return_any():
    return reflect(on=None, skip=True)


def api_base_return_any():
    return reflect(reset=True)


def api_base_return_array(input_arg=0):
    return reflect(on="self" if input_arg is None else input_arg)


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
