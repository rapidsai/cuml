# Copyright (c) 2025, NVIDIA CORPORATION.

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.array import CumlArray
from cuml.internals.memory_utils import using_output_type
from cuml.internals.input_utils import (
    determine_array_type,
    determine_array_dtype,
)

import inspect
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import cupy as cp
import numpy as np

# Everything related to API boundary determination


def in_internal_api() -> bool:
    return GlobalSettings().api_depth > 1


def cuml_public_api(func):
    @wraps(func)
    def inner(*args, **kwargs):
        GlobalSettings().increment_api_depth()
        try:
            return func(*args, **kwargs)
        finally:
            GlobalSettings().decrement_api_depth()

    return inner


# CumlArray


def as_cuml_array(X, dtype=None) -> CumlArray:
    """Wraps array X in CumlArray container."""
    # TODO: After debugging, replace this with an immediate call to the CumlArray constructor.
    return CumlArray(X, dtype=dtype)


# CumlArrayDescriptor
# TODO: Replace CumlArrayDescriptor in cuml/common/array_descriptor.py


class CumlArrayDescriptor:
    def __init__(self, order="K", dtype=None):
        self.order = order
        self.dtype = dtype

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        # Save the provided value as CumlArray and initialize output cache.
        dtype = self.dtype or _get_dtype(obj)
        setattr(
            obj,
            f"_{self.name}_data",
            None if value is None else as_cuml_array(value, dtype),
        )
        setattr(obj, f"_{self.name}_output_cache", dict())

    def _to_cached_output(self, obj, array, output_type):
        output_cache = getattr(obj, f"_{self.name}_output_cache")

        if output_type not in output_cache:
            output_cache[output_type] = array.to_output(output_type)

        return output_cache[output_type]

    @cuml_public_api
    def __get__(self, obj, _=None):

        # The descriptor was accessed on a class rather than an instance.
        if obj is None:
            return self

        # Get data from the owning object
        array = getattr(obj, f"_{self.name}_data")

        # This is accessed internally, just return the cuml array directly.
        if in_internal_api():
            return array

        # The global output type is set, return the array converted to that.
        elif (global_output_type := GlobalSettings().output_type) is not None:
            return self._to_cached_output(obj, array, global_output_type)

        # Return the array converted to the object's _output_type
        elif (output_type := getattr(obj, "_output_type", None)) is not None:
            return self._to_cached_output(obj, array, output_type)

        # Neither the global nor the object's output_type are set. Since this
        # is a user call, we must fail.
        else:
            raise RuntimeError(
                "Tried to access CumlArrayDescriptor without output_type set."
            )

    def __delete__(self, obj):
        if obj is not None:
            delattr(obj, f"_{self.name}_data")


# Type reflection


def _set_output_type(obj: Any, output_type: str):
    setattr(obj, "_output_type", output_type)


def _set_dtype(obj: Any, dtype):
    setattr(obj, "dtype", dtype)


def _get_dtype(obj: Any):
    return getattr(obj, "dtype", None)


class set_output_type:
    """Set a object's output_type based on a function argument type.

    Example:

        @set_output_type("X")
        def fit(self, X, y):
             ...

    Sets the output_type of self to the type of the X argument.
    """

    def __init__(self, to, dtype=None):
        if isinstance(to, str):
            to = TypeOfArgument(to)

        self.to = to
        self.dtype = dtype

    def __call__(self, func):
        sig = inspect.signature(func)

        @wraps(func)
        @cuml_public_api
        def inner(obj, *args, **kwargs):
            if not in_internal_api():
                bound_args = sig.bind(obj, *args, **kwargs)
                bound_args.apply_defaults()

                if isinstance(self.to, TypeOfArgument):
                    arg_value = bound_args.arguments.get(self.to.argument_name)
                    arg_type = determine_array_type(arg_value)
                    if arg_type is None:
                        raise TypeError(
                            f"Argument for {self.to.argument_name} must be array-like."
                        )
                    dtype = self.dtype or determine_array_dtype(arg_value)
                    _set_output_type(obj, arg_type)
                    _set_dtype(obj, dtype)
                else:
                    raise TypeError(
                        f"Cannot handle self.to type '{type(self.to)}."
                    )

            return func(obj, *args, **kwargs)

        return inner


def _to_output_type(obj, output_type: str):
    """Convert CumlArray and containers of CumlArray."""
    if isinstance(obj, CumlArray):
        return obj.to_output(output_type)
    elif isinstance(obj, Sequence) and not isinstance(obj, str):
        return type(obj)(_to_output_type(item) for item in obj)
    else:
        return obj


# Sentinels
ObjectOutputType = object()

DefaultOutputType = ObjectOutputType


@dataclass
class TypeOfArgument:
    argument_name: str


class convert_cuml_arrays:
    def __init__(self, to=DefaultOutputType):
        self.to = to

    def __call__(self, func):
        sig = inspect.signature(func)

        @wraps(func)
        @cuml_public_api
        def inner(*args, **kwargs):
            ret = func(*args, **kwargs)

            # Internal call, just return the value without further processing.
            if in_internal_api():
                return ret

            # We use the global output type, whenever it is set.
            elif (
                global_output_type := GlobalSettings().output_type
            ) is not None:
                return _to_output_type(ret, global_output_type)

            # Use the object's output type, assumes that func is a method with self argument.
            elif self.to is ObjectOutputType:
                # Use the object's output type.
                obj = args[0]
                output_type = obj._output_type

            elif isinstance(self.to, TypeOfArgument):
                # Use the type of the function argument.
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                arg_value = bound_args.arguments.get(self.to.argument_name)
                output_type = determine_array_type(arg_value)
            else:
                raise ValueError(f"Unable to process 'to' argument: {self.to}")

            return _to_output_type(ret, output_type)

        return inner
