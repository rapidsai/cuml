# Copyright (c) 2025, NVIDIA CORPORATION.

import inspect
import numpy as np
import cupy as cp
from dataclasses import dataclass
from functools import wraps
from collections import OrderedDict
from typing import Any

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
                return np.asarray(self.data)
            case "cupy":
                return cp.asarray(self.data)
            case _:
                raise TypeError(f"Unknown output_type '{output_type}'.")


## CumlArrayDescriptor

class CumlArrayDescriptor:

    def __init__(self, order="K"):
        self.order = order

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, obj, value):
        # Just save the provided value as CumlArray
        setattr(obj, f"_{self.name}_value", CumlArray(value))

    def __get__(self, obj, objtype=None):
        value = getattr(obj, f"_{self.name}_value")
        if global_api_counter > 0:
            return value
        else:
            output_type = _get_output_type(obj)
            return value.to_output(output_type)


## Type reflection

global_output_type = None

def determine_array_type(value) -> str:
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

def _get_output_type(obj: Any):
    if global_output_type is None:
        return getattr(obj, "_output_type", None)
    else:
        return global_output_type


class set_output_type:  # decorator
    
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


def to_output_type(return_value, output_type: str):
    """Convert CumlArray and containers of CumlArray."""
    if type(return_value) is CumlArray:
        return return_value.to_output(output_type)
    elif type(return_value) is tuple:
        return tuple(to_output_type(item) for item in return_value)
    else:
        return return_value


def convert_cuml_arrays(func):  # decorator

    @wraps(func)
    @api_boundary
    def inner(obj, *args, **kwargs):
        ret = func(obj, *args, **kwargs)
        if is_api_internal():
            return ret
        else:
            output_type = _get_output_type(obj)
            return to_output_type(ret, output_type)

    return inner

## Example estimator implementation

class MinimalLinearRegression:

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    @set_output_type("X")
    def fit(self, X, y):
        X = CumlArray(X).to_output("numpy")
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute coefficients using normal equation
        weights = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y

        # Separate intercept and coefficients
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]

        return self

    @convert_cuml_arrays
    def predict(self, X):
        X = CumlArray(X).to_output("numpy")
        y = X @ self.coef_.to_output("numpy") + self.intercept_.to_output("numpy")
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
