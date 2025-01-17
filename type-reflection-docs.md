# Revised Type Reflection System

Author: @csadorf

## Motivation

The purpose of the type reflection system is to enable users to provide inputs
and parameters in their preferred format, e.g., numpy or cupy arrays, or pandas
DataFrame objects and have results be returned in the same or format without
needing to worry about any internal processing and conversion of the data.

For this purpose, estimators will _reflect_ the provided type in most cases.

## Use cases

### Data transformation on simple estimator

Here we assume that a user wants to train a linear regression model on input
data (_X_, _y_) provided in the form of numpy arrays and expects the prediction
result to be returned in the form of a numpy array as well. This is how cuml
would support this:

```python
# Instantiate and train the estimator
model = MinimalLinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

The type reflection system would guarantee that the following assert holds true:
```python
assert type(predictions) == type(X)
```

## Development

### Estimator Development

Developing a cuml estimator class that uses the reflection system requires three main components:

1. The specification of how the desired output type is determined.
2. The specification of which functions should reflect the type and ensuring that to be converted arrays are returned as `CumlArray` type.
3. Specyifying all class attributes that should reflect type are declared as `CumlArrayDescriptor` types.

Here is a minimal example skipping the actual implementation:

```python
class MinimalLinearRegression:

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    @set_output_type("X")
    def fit(self, X, y):
        ...
        return self

    @convert_cuml_arrays()
    def predict(self, X):
        ...
        return CumlArray(y)
```

In this case we declared both attributes `coef_` and `intercept_` to be of type
`CumlArrayDescriptor` type which means that they will be automatically converted
to their owner's `output_type` unless the global `output_type` is set.

The `fit()` method is decorated with the `@set_output_type("X")` decorator which
means that the object's `output_type` should be set to the method's "X" argument
type.

The `predict()` method is decorated with the `@convert_cuml_arrays()` decorator
which means that `CumlArrays` returned from this function are converted to the
object's `output_type`.

### Internal vs. external calls

The type reflection system ensures type consistency for users, however type
conversions should otherwise be avoided to minimize the number of host-device
data transfers. For example, when one cuml estimator calls another cuml
transformer internally, the data should only be copied at the final step when it
is returned to the user.

To achieve this, we keep track of whether a cuml API call was made externally at
the user-level, or internally. A developer can always check the current API
stack level with the `in_internal_api()` function.

The `convert_cuml_arrays` decorator will only trigger conversions for external
API calls, right before data is handed back to the user.

### The global output type

It is possible to override the dynamic output type by setting the global output type.
Example:

```python
with using_output_type("cupy"):
    ...
```

All outputs within this context will be converted to cupy arrays regardless of
any other configuration.

Note: It is **not** possible to opt out of the global output type override. If a
function needs to return a specific type regardless of the global output type
you cannot use the `@convert_cuml_arrays` decorator. This is to ensure
behavioral consistency across the cuml API for both users and developers.

## Advanced conversion

The default behavior of the `convert_cuml_arrays()` decorator is to convert cuml arrays

1. To the global output type if set.
2. The object's output type.

_The function will fail if neither is set._

The behavior can be modified by setting the `to` argument:

```
@convert_cuml_arrays()  # default behavior
# equivalent to
@convert_cuml_arrays(to=DefaultOutputType)

If you want to use the 

# Use the type of the argument named "X":
# @convert_cuml_arrays(to=(DefaultOutputType, TypeOfArgument("X")))

# Always use the globally set output type:
# @convert_cuml_arrays(to=GlobalOutputType)
```
