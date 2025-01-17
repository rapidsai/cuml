# Copyright (c) 2025, NVIDIA CORPORATION.

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.array import CumlArray

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


# # CumlArray


def as_cuml_array(X) -> CumlArray:
    """Wraps array X in CumlArray container."""
    return CumlArray(X)


# CumlArrayDescriptor


class CumlArrayDescriptor:
    def __init__(self, order="K"):
        self.order = order

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        # Just save the provided value as CumlArray
        setattr(obj, f"_{self.name}_data", as_cuml_array(value))

    @cuml_public_api
    def __get__(self, obj, _=None):

        if (
            obj is None
        ):  # descriptor was accessed on class rather than instance
            return self

        # Get data from the owning object
        array = getattr(obj, f"_{self.name}_data")

        # This is accessed internally, just return the cuml array directly.
        if in_internal_api():
            return array

        # The global output type is set, return the array converted to that.
        elif (global_output_type := GlobalSettings().output_type) is not None:
            return array.to_output(global_output_type)

        # Return the array converted to the object's _output_type
        elif (output_type := obj._output_type) is not None:
            return array.to_output(output_type)

        # Neither the global nor the object's output_type are set. Since this
        # is a user call, we must fail.
        else:
            raise RuntimeError(
                "Tried to access CumlArrayDescriptor without output_type set."
            )


# Type reflection


@contextmanager
def override_output_type(output_type: str):
    try:
        previous_output_type = GlobalSettings().output_type
        GlobalSettings().output_type = output_type
        yield
    finally:
        GlobalSettings().output_type = previous_output_type


def determine_array_type(value) -> str:
    """Utility function to identify the array type."""
    if isinstance(value, CumlArray):
        return "cuml"
    elif isinstance(value, np.ndarray):
        return "numpy"
    elif isinstance(value, cp.ndarray):
        return "cupy"
    else:
        return ValueError(f"Unknown array type: {type(value)}")


def _set_output_type(obj: Any, output_type: str):
    setattr(obj, "_output_type", output_type)


class set_output_type:
    """Set a object's output_type based on a function argument type.

    Example:

        @set_output_type("X")
        def fit(self, X, y):
             ...

    Sets the output_type of self to the type of the X argument.
    """

    def __init__(self, to):
        if isinstance(to, str):
            to = TypeOfArgument(to)

        self.to = to

    def __call__(self, func):
        sig = inspect.signature(func)

        @cuml_public_api
        def inner(obj, *args, **kwargs):
            if not in_internal_api():
                bound_args = sig.bind(obj, *args, **kwargs)
                bound_args.apply_defaults()

                if isinstance(self.to, TypeOfArgument):
                    arg_value = bound_args.arguments.get(self.to.argument_name)
                    arg_type = determine_array_type(arg_value)
                    _set_output_type(obj, arg_type)
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
GlobalOutputType = object()

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


# Example estimator implementation


class MinimalLinearRegression:

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    # Private methods should not be at the API boundary and must
    # never use the @set_output_type decorator.

    def _fit_on_device(self, X: cp.ndarray, y: cp.ndarray):
        X_design = cp.hstack([cp.ones((X.shape[0], 1)), X])

        # Compute coefficients using normal equation
        weights = cp.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        # Separate intercept and coefficients
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

    @set_output_type("X")
    def fit(self, X, y):
        # The implementation here is device specific. We delay the conversion to
        # CumlArray and then device array to the latest possible moment.
        X, y = as_cuml_array(X), as_cuml_array(y)
        self._fit_on_device(X.to_device_array(), y.to_device_array())

        return self

    def _predict_on_device(self, X: cp.ndarray) -> cp.ndarray:
        # This is an API internal method, the array descriptor will not(!)
        # perform an automatic conversion.
        return (
            X @ self.coef_.to_device_array()
            + self.intercept_.to_device_array()
        )

    @convert_cuml_arrays(to=ObjectOutputType)
    def predict(self, X):
        y = self._predict_on_device(as_cuml_array(X).to_device_array())

        # By returning the result within the CumlArray container in a function
        # at the API boundary decorated with @convert_cuml_arrays, we ensure
        # that the return value is automatically converted to reflect the desired
        # type.
        return CumlArray(y)


def example_workflow():
    # Example usage
    from sklearn.datasets import make_regression

    # Create synthetic data
    X, y = make_regression(
        n_samples=20, n_features=2, noise=0.1, random_state=42
    )

    # Instantiate and train the estimator
    model = MinimalLinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions, type(predictions))
    print("coef", model.coef_, type(model.coef_))

    with override_output_type("cupy"):
        assert isinstance(model.coef_, cp.ndarray)

    assert isinstance(model.coef_, np.ndarray)

    # Example for reflection of types of a stateless function.
    @convert_cuml_arrays(to=TypeOfArgument("X"))
    def power(X, exponent: int):
        X = as_cuml_array(X)
        result = cp.sqrt(X.to_device_array())
        return as_cuml_array(result)

    squared_X = power(X, 2)
    assert isinstance(squared_X, type(X))

    with override_output_type("cupy"):
        assert isinstance(power(X, 2), cp.ndarray)


if __name__ == "__main__":
    example_workflow()
