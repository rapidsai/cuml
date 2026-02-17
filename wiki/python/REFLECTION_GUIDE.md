# Reflection System Developer Guide

This guide covers the `reflect`, `run_in_internal_context`, and `exit_internal_context` utilities from `cuml.internals.outputs`. These provide a streamlined approach to output type management that replaces the older `@api_base_return_*` decorators for many use cases.

## Overview

cuML's reflection system manages output type conversions, ensuring arrays are returned in the user's expected format (cupy, numpy, pandas, cudf, etc.). The system tracks "internal" vs "external" contexts to avoid unnecessary conversions during intermediate computations.

**Key Concepts:**

- **External context**: Code called directly by the user. Arrays should be converted to the requested output type.
- **Internal context**: Code called within cuML methods. Arrays remain as `CumlArray` to avoid conversion overhead.

## `@reflect` Decorator

Use `@reflect` on methods/functions that return arrays to the user. This is the primary decorator for most estimator methods.

### When to Use

- **Fit methods** that set model attributes: use `@reflect(reset=True)`
- **Transform/predict methods** that return array outputs: use `@reflect`
- **Property accessors** that return arrays: use `@reflect`
- **Standalone functions** that return arrays: use `@reflect`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reset` | `False` | Set `True` for fit-like methods. Stores `_input_type` and `n_features_in_` on the model. |
| `model` | `'self'` if present | Parameter name/index for the estimator instance. Set to `None` for standalone functions. |
| `array` | First non-model arg | Parameter name/index to infer input type from. Set to `None` for methods with no array input. |

### Examples

```python
from cuml.internals import reflect
from cuml.internals.array import CumlArray
from cuml.internals.base import Base

class MyEstimator(Base):
    # Fit method: reset=True stores input metadata
    @reflect(reset=True)
    def fit(self, X, y=None):
        self.X_ = CumlArray.from_input(X)
        return self

    # Standard method returning an array
    @reflect
    def transform(self, X):
        return some_computation(X)

    # Property returning an array
    @property
    @reflect
    def components_(self):
        return self._components

    # Method with no array input (uses fit-time input type)
    @reflect
    def get_components(self):
        return self._components


# Standalone function
@reflect
def my_function(X):
    return process(X)

# Function where first argument is not an array
@reflect(array=None)
def forecast(nsteps):
    return generate_forecast(nsteps)
```

### Behavior

1. Enters an internal context (nested reflected calls return `CumlArray`)
2. If `reset=True`, stores `_input_type` and `n_features_in_` from the array parameter
3. Executes the wrapped function
4. Converts output arrays to the appropriate type based on:
   - Global `output_type` setting (highest priority)
   - Estimator's `output_type` attribute
   - Input array type (if `output_type="input"`)
   - Default: `cupy`

### Nested Output Conversion

The `@reflect` decorator automatically handles nested structures. Arrays within tuples, lists, and dicts are all converted:

```python
@reflect
def predict_with_bounds(self, X):
    predictions = compute_predictions(X)
    lower = compute_lower_bound(X)
    upper = compute_upper_bound(X)
    # All three arrays are converted to the appropriate output type
    return predictions, lower, upper
```

## `@run_in_internal_context` Decorator

Use `@run_in_internal_context` for methods that:
- Call other reflected methods internally
- Need to work with `CumlArray` objects
- **Do not** participate in automatic output conversion

### When to Use

- Methods that call reflected methods and handle output conversion manually
- Internal helper methods that aggregate results from other methods
- Score methods that return scalars (not arrays)
- Methods that need to call `decode_labels` or similar post-processing

### Example

```python
from cuml.internals import run_in_internal_context, exit_internal_context

class MyClassifier(Base):
    @run_in_internal_context
    def predict(self, X):
        # Call reflected method - returns CumlArray internally
        scores = self.decision_function(X)

        # Manual processing on internal types
        indices = (scores.to_output("cupy") >= 0).view(cp.int8)

        # Manual output conversion using exit_internal_context
        with exit_internal_context():
            output_type = self._get_output_type(X)
        return decode_labels(indices, self.classes_, output_type=output_type)

    @run_in_internal_context
    def score(self, X, y):
        # Returns a scalar, not an array - no reflection needed
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
```

### Key Difference from `@reflect`

| Aspect | `@reflect` | `@run_in_internal_context` |
|--------|------------|---------------------------|
| Output conversion | Automatic | Manual |
| Sets `_input_type` | With `reset=True` | No |
| Enters internal context | Yes | Yes |
| Use case | Array-returning methods | Helper methods, score methods |

## `exit_internal_context` Context Manager

Use `exit_internal_context()` to temporarily exit an internal context when you need to:
- Call external APIs that might be reflected
- Get the correct output type for manual conversion
- Call other estimator methods that should behave as if called externally

### When to Use

- Inside `@run_in_internal_context` methods when manually converting output
- When calling scikit-learn or other external estimators that may use reflection
- When calling `self._get_output_type(X)` for manual type inference

### Example

```python
from cuml.internals import run_in_internal_context, exit_internal_context

class MySVC(Base):
    @run_in_internal_context
    def predict(self, X):
        scores = self.decision_function(X)
        indices = cp.argmax(scores.to_output("cupy"), axis=1)

        # Exit internal context to get the user-facing output type
        with exit_internal_context():
            output_type = self._get_output_type(X)

        return decode_labels(indices, self.classes_, output_type=output_type)

    @reflect(reset=True)
    def fit(self, X, y):
        if self.probability:
            # Exit to call external calibration API that may use reflection
            with exit_internal_context():
                calibrator.fit(X, y)
        # ...
        return self
```

## Decision Flowchart

```
Does the method return array(s) to the user?
├── Yes → Does it need special post-processing (decode_labels, etc.)?
│   ├── Yes → Use @run_in_internal_context + exit_internal_context()
│   └── No  → Use @reflect
│            └── Is it a fit-like method? → Add reset=True
└── No  → Does it call reflected methods internally?
    ├── Yes → Use @run_in_internal_context
    └── No  → No decorator needed
```

## Common Patterns

### Classifier `predict` with Label Decoding

When `predict` needs to decode numeric indices back to class labels:

```python
@run_in_internal_context
def predict(self, X):
    scores = self.decision_function(X)  # Returns CumlArray
    indices = process_scores(scores)

    with exit_internal_context():
        output_type = self._get_output_type(X)
    return decode_labels(indices, self.classes_, output_type=output_type)
```

### Method with No Input Array

For methods like `forecast` where there's no array to infer the output type from:

```python
@reflect(array=None)
def forecast(self, nsteps):
    # Uses fit-time input type for output conversion
    return self._generate_forecast(nsteps)
```

### Score Method Returning a Scalar

Score methods return floats, not arrays, so use `run_in_internal_context`:

```python
@run_in_internal_context
def score(self, X, y):
    predictions = self.predict(X)
    return accuracy_score(y, predictions)
```

### Property Returning an Array

```python
@property
@reflect
def support_(self):
    return self._support_vectors
```

### Calling External Estimators

When calling external code that might participate in reflection:

```python
@reflect(reset=True)
def fit(self, X, y):
    with exit_internal_context():
        # CalibratedClassifierCV may use reflected methods internally
        cccv = CalibratedClassifierCV(base_estimator)
        cccv.fit(X, y)
    self._calibrator = cccv
    return self
```

## Relationship to Older Decorators

The `@reflect` decorator is the modern replacement for the `@api_base_return_array` family of decorators. Key differences:

| Old Pattern | New Pattern |
|-------------|-------------|
| `@api_base_return_array()` with return type `-> CumlArray` | `@reflect` |
| `@api_base_return_any(set_output_type=True, set_n_features_in=True)` | `@reflect(reset=True)` |
| Manual `_set_output_type()` + `_get_output_type()` | Automatic via `@reflect` |
| `with cuml.using_output_type("mirror"):` | Automatic via `@reflect` entering internal context |

Both systems coexist in the codebase. New code should prefer `@reflect` for its simpler API.

## Testing

When testing reflected methods, verify output types work correctly:

```python
def test_output_type():
    X = rand_array("numpy")
    model = MyEstimator().fit(X)

    # Default: reflects input type
    assert isinstance(model.transform(X), np.ndarray)

    # Global override
    with cuml.using_output_type("cudf"):
        assert isinstance(model.transform(X), cudf.DataFrame)

    # Estimator override
    model.output_type = "pandas"
    assert isinstance(model.transform(X), pd.DataFrame)
```
