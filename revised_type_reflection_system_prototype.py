# Copyright (c) 2025, NVIDIA CORPORATION.

import inspect
import numpy as np
import cupy as cp
from dataclasses import dataclass
from functools import wraps
from collections import OrderedDict
from typing import Any
from collections.abc import Sequence

## Everything related to API boundary determination

global_api_counter : int = 0

def is_api_internal():
    return global_api_counter > 1


def api_boundary(func):

    @wraps(func)
    def inner(*args, **kwargs):
        global global_api_counter
        global_api_counter += 1
        try:
            return func(*args, **kwargs)
        finally:
            global_api_counter -= 1

    return inner


## CumlArray

class CumlArray:

    def __init__(self, data):
        self.data = data

    def to_output(self, output_type: str):
        match output_type:
            case  "numpy":
                if isinstance(self.data, cp.ndarray):
                    return self.data.get()
                else:
                    return np.asarray(self.data)
            case "cupy":
                return cp.asarray(self.data)
            case _:
                raise TypeError(f"Unknown output_type '{output_type}'.")

    def to_device_array(self) -> cp.ndarray:
        return self.to_output("cupy")


def as_cuml_array(X) -> CumlArray:
    """Wraps array X in CumlArray container."""
    return CumlArray(X)


## CumlArrayDescriptor

class CumlArrayDescriptor:

    def __init__(self, order="K"):
        self.order = order

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        # Just save the provided value as CumlArray
        setattr(obj, f"_{self.name}_value", as_cuml_array(value))

    def __get__(self, obj, objtype=None):
        # Return either the original value for internal access or convert to the
        # desired output type.
        value = getattr(obj, f"_{self.name}_value")
        if global_api_counter > 0:
            return value
        else:
            output_type = _get_output_type(obj)
            return value.to_output(output_type)


## Type reflection

global_output_type = None

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

def _get_output_type(obj: Any) -> str:
    if global_output_type is None:
        return getattr(obj, "_output_type", None)
    else:
        return global_output_type


class set_output_type:
    """Set a object's output_type based on a function argument type.

    Example:

        @set_output_type("X")
        def fit(self, X, y):
             ...

    Sets the output_type of self to the type of the X argument.
    """
    
    def __init__(self, arg_name: str):
        self.arg_name = arg_name

    def __call__(self, func):
        sig = inspect.signature(func)

        @api_boundary
        def inner(obj, *args, **kwargs):
            if not is_api_internal():
                bound_args = sig.bind(obj, *args, **kwargs)
                bound_args.apply_defaults()

                arg_value = bound_args.arguments.get(self.arg_name)
                arg_type = determine_array_type(arg_value)
                _set_output_type(obj, arg_type)

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


def convert_cuml_arrays(func):  # decorator
    """Cuml arrays in method return value are converted."""

    @wraps(func)
    @api_boundary
    def inner(obj, *args, **kwargs):
        ret = func(obj, *args, **kwargs)
        if is_api_internal():
            return ret
        else:
            output_type = _get_output_type(obj)
            return _to_output_type(ret, output_type)

    return inner

## Example estimator implementation

class MinimalLinearRegression:

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    # Private methods should not be at the API boundary and should
    # not use the @set_output_type decorator.

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
        return X @ self.coef_.to_device_array() + self.intercept_.to_device_array()

    @convert_cuml_arrays
    def predict(self, X):
        y = self._predict_on_device(as_cuml_array(X).to_device_array())

        # By returning the result within the CumlArray container in a function
        # at the API boundary decorated with @convert_cuml_arrays, we ensure
        # that the return value is automatically converted to reflect the desired
        # type.
        return CumlArray(y)


def test():
    # Example usage
    from sklearn.datasets import make_regression

    # Create synthetic data
    X, y = make_regression(n_samples=20, n_features=2, noise=0.1, random_state=42)

    # Instantiate and train the estimator
    model = MinimalLinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions, type(predictions))
    print("coef", model.coef_, type(model.coef_))


if __name__ == "__main__":
    test()
