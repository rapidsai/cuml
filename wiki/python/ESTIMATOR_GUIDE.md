# cuML Python Estimators Developer Guide

This guide is meant to help developers follow the correct patterns when creating/modifying any cuML Estimator object and ensure a uniform cuML API.

**Note:** Start with the [Quick Start Guide](#quick-start-guide) and [copyable estimator skeleton](#copyable-estimator-skeleton). The later sections explain the estimator contract in more detail for less common cases.

## Table of Contents

- [Recommended Scikit-Learn Documentation](#recommended-scikit-learn-documentation)
- [API Matching Policy](#api-matching-policy)
- [Quick Start Guide](#quick-start-guide)
   - [Copyable Estimator Skeleton](#copyable-estimator-skeleton)
- [Background](#background)
   - [Array I/O and Output Types in cuML](#array-io-and-output-types-in-cuml)
   - [Ingesting Arrays](#ingesting-arrays)
   - [Returning Arrays](#returning-arrays)
- [Estimator Design](#estimator-design)
   - [Initialization](#initialization)
   - [Implementing `_get_param_names()`](#implementing-_get_param_names)
   - [Estimator Tags and cuML Specific Tags](#estimator-tags-and-cuml-specific-tags)
   - [Estimator Array-Like Attributes](#estimator-array-like-attributes)
   - [Estimator Methods](#estimator-methods)
- [Do's and Do Not's](#dos-and-do-nots)
- [Appendix](#appendix)

## Recommended Scikit-Learn Documentation

To start, it's recommended to read the following Scikit-learn documentation:

1. [Scikit-learn's Estimator Docs](https://scikit-learn.org/stable/developers/develop.html)
   1. cuML Estimator design follows Scikit-learn very closely. We will only cover portions where our design differs from this document
   2. If short on time, pay attention to these sections, which are the most important (and have caused pain points in the past):
      1. [Instantiation](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      2. [Estimated Attributes](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      3. [`get_params` and `set_params`](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      4. [Cloning](https://scikit-learn.org/stable/developers/develop.html#cloning)
      5. [Estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
2. [Scikit-learn's Docstring Guide](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation)
   1. We follow the same guidelines for specifying array-like objects, array shapes, dtypes, and default values

## API Matching Policy

cuML often implements GPU-accelerated versions of estimators that already exist in CPU-based libraries, with scikit-learn being the most prominent example. When implementing these variants, we aim to maintain API compatibility with the original library while following these guidelines:

1. **Match the API of the original library where possible and reasonable**
   - Use identical parameter names, types, and default values
   - Keep method signatures and return types consistent
   - Maintain the same behavior and semantics where possible

2. **API deviations must be well-justified and documented**
   - Document all API deviations clearly
   - Avoid arbitrary deviations from the original API

     For example, if the original library uses a parameter named `n_neighbor`, we should not arbitrarily change it to `n_neighbors` in our implementation.

   - Explain necessary deviations with in-code comments

3. **Unused parameters or arguments should generally not be matched**
   - If a parameter exists in the original API but isn't used in cuML's implementation, it's better to omit it
   - This helps maintain a cleaner, more focused API and is less surprising to both users and developers
   - However, consider backwards compatibility before removing unused parameters that are already present

4. **Exact API matching is not required**
   - Consumers who need exact API matching should use `cuml.accel`
   - Focus on providing a consistent and intuitive API rather than exact matching
   - Prioritize performance and GPU-specific optimizations over exact API matching
   - To emphasize: while exact API matching is not required, arbitrary deviations are not permitted

## Quick Start Guide

At a high level, all cuML Estimators must:
1. Inherit from `cuml.Base`
   ```python
   import cuml

   class MyEstimator(cuml.Base):
      ...
   ```
2. Follow the Scikit-learn estimator guidelines found [here](https://scikit-learn.org/stable/developers/develop.html)
3. Include the `Base.__init__()` arguments available in the new Estimator's `__init__()`
   ```python
   import cuml

   class MyEstimator(cuml.Base):

      def __init__(self, *, extra_arg=True, verbose=False, output_type=None):
         super().__init__(verbose=verbose, output_type=output_type)
         ...
   ```

   > **Note:** The `handle` argument has been removed from `Base.__init__`. New estimators should not include a `handle` parameter. If your estimator requires `n_streams` or multi-GPU support via `device_ids`, add those as top-level parameters instead.
4. Declare each array-like attribute the new Estimator will compute as a class variable for automatic array type conversion. An order can be specified to serve as an indicator of the order the array should be in for the C++ algorithms to work.
   ```python
   import cuml
   from cuml.common.array_descriptor import CumlArrayDescriptor

   class MyEstimator(cuml.Base):

      labels_ = CumlArrayDescriptor(order='C')

      def __init__(self):
         ...
   ```
5. Use the `@reflect` decorator on public API methods that return arrays. Use `@reflect(reset=True)` for simple fit-like methods, or `@reflect(reset="type")` if the method handles feature validation itself:
   ```python
   import cuml
   from cuml.internals import reflect
   from cuml.internals.array import CumlArray

   class MyEstimator(cuml.Base):

      @reflect(reset=True)
      def fit(self, X) -> "MyEstimator":
         ...

      @reflect
      def predict(self, X) -> CumlArray:
         ...
   ```
   See the [Reflection Guide](REFLECTION_GUIDE.md) for detailed guidance on when to use `@reflect`, `@run_in_internal_context`, and `exit_internal_context`.
6. Implement `_get_param_names()` including values returned by `super()._get_param_names()`
   ```python
      @classmethod
      def _get_param_names(cls):
         return super()._get_param_names() + [
            "eps",
            "min_samples",
         ]
   ```

7. Implement the appropriate tags method if any of the [default tags](#estimator-tags-and-cuml-specific-tags) need to be overridden for the new estimator.
There are some convenience [Mixins](../../python/cuml/cuml/internals/mixins.py), that the estimator can inherit, can be used for indicating the preferred order (column or row major) as well as for sparse input capability.

If other tags are needed, they are static (i.e. don't change depending on the instantiated estimator), and more than one estimator will use them, then implement a new [Mixin](../../python/cuml/cuml/internals/mixins.py), if the tag will be used by a single class then implement the `_more_static_tags` method:
   ```python
    @staticmethod
    def _more_static_tags():
       return {
            "requires_y": True
       }
   ```
   If the tags depend on an attribute that is defined at runtime or instantiation of the estimator, then implement the `_more_tags` method:
   ```python
      def _more_tags(self):
           return {
               "allow_nan": is_scalar_nan(self.missing_values)
            }
   ```

For the majority of estimators, the above steps will be sufficient to correctly work with the cuML library and ensure a consistent API. However, situations may arise where an estimator differs from the standard pattern and some of the functionality needs to be customized. The remainder of this guide takes a deep dive into the estimator functionality to assist developers when building estimators.

### Copyable Estimator Skeleton

Use this as a starting point for dense estimators that follow the standard cuML pattern:

```python
import cuml
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals import reflect
from cuml.internals.array import CumlArray


class MyEstimator(cuml.Base):
    result_ = CumlArrayDescriptor(order="C")

    def __init__(self, *, extra_arg=True, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.extra_arg = extra_arg

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ["extra_arg"]

    @reflect(reset=True)
    def fit(self, X, y=None) -> "MyEstimator":
        X_m = input_to_cuml_array(X, order="K").array
        # Replace this placeholder with estimator training.
        self.result_ = X_m
        return self

    @reflect
    def transform(self, X) -> CumlArray:
        X_m = input_to_cuml_array(X, order="K").array
        # Return an array-like object directly; @reflect handles conversion.
        return X_m
```

For fit-like methods that call validation directly, use `@reflect(reset="type")` and let the validation helper set or check `n_features_in_`. Methods that return scalars usually do not need `@reflect`; use `@run_in_internal_context` only when the method calls reflected methods internally.

## Background

Some background is necessary to understand the design of estimators and how to work around any non-standard situations.

### Array I/O and Output Types in cuML

cuML estimators should accept the standard array-like inputs supported by `input_to_cuml_array`: cuDF DataFrame or Series, pandas DataFrame or Series, NumPy arrays, Numba device arrays, CuPy arrays, and internal `CumlArray` values. Sparse estimators should use the sparse-specific validation utilities used by neighboring sparse estimators.

Internally, dense array data should usually be converted to `CumlArray`. `CumlArray` can store either device-accessible or host-accessible data. Low-level code that needs a specific memory location should request it explicitly through arguments such as `convert_to_mem_type` on input conversion or `output_mem_type` on `CumlArray.to_output()`.

Public output type conversion is handled by `@reflect` and `CumlArrayDescriptor`. Users can choose output types in three ways:

1. Set `output_type` on an estimator, for example `MyEstimator(output_type="numpy")`.
2. Set a global override with `cuml.set_global_output_type("numpy")`.
3. Temporarily set a global override with `cuml.using_output_type("numpy")`.

The global setting stored in `cuml.global_settings.output_type` takes precedence over an estimator's `output_type`. When neither is set, reflected estimator methods normally mirror the call input type, and descriptor attributes mirror the fit-time input type.

Accepted output type strings are:

 - `None`: No global or estimator override. Reflected estimator methods infer from their input or fit-time input type.
 - `"input"`: Mirror the relevant input type.
 - `"array"`: Return a CuPy array for device data or a NumPy array for host data.
 - `"numba"`: Return a Numba device array.
 - `"dataframe"`: Return a cuDF or pandas DataFrame based on memory location.
 - `"series"`: Return a cuDF or pandas Series based on memory location.
 - `"df_obj"`: Return a Series for single-dimensional output or a DataFrame otherwise.
 - `"cupy"`: Return a CuPy array.
 - `"numpy"`: Return a NumPy array.
 - `"cudf"`: Return a cuDF Series or DataFrame.
 - `"pandas"`: Return a pandas Series or DataFrame.

The internal output type `"cuml"` may appear inside reflected calls. User-facing code should not set it.

### Ingesting Arrays

When the input array type isn't known, the correct and safest way to ingest arrays is using `cuml.common.input_to_cuml_array`. This method can handle all supported types, is capable of checking the array order, can enforce a specific dtype, and can raise errors on incorrect array sizes:

```python
def fit(self, X):
    input_data = input_to_cuml_array(X, order="K")
    X_m = input_data.array
    rows = input_data.n_rows
    cols = input_data.n_cols
    dtype = input_data.dtype
    ...
```

### Returning Arrays

The `CumlArray` class can convert to any supported array type using the `to_output(output_type: str)` method. However, doing this explicitly is almost never needed in practice and **should be avoided**. Directly converting arrays with `to_output()` will circumvent the automatic conversion system potentially causing extra or incorrect array conversions.

## Estimator Design

All estimators (any class that is a child of `cuml.Base`) have a similar structure. In addition to the guidelines specified in the [SkLearn Estimator Docs](https://scikit-learn.org/stable/developers/develop.html), cuML implements a few additional rules.

### Initialization

All estimators should match the arguments (including the default value) in `Base.__init__` and pass these values to `super().__init__()`. All estimators should accept `verbose` and `output_type`.

> **Note:** The `handle` argument has been removed from `Base.__init__`. New estimators should not include a `handle` parameter. Estimators that need to configure stream pools should add an `n_streams` parameter. Estimators that support multi-GPU execution should add a `device_ids` parameter.

In general, all estimator constructor parameters should be keyword-only except for those arguments that are not keyword-only in the matched API. This helps prevent breaking changes if arguments are added or removed in future versions. For example:

```python
# For an estimator that matches scikit-learn's API where the eps argument can be positional:
def __init__(self, eps=0.5, *, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, verbose=False, output_type=None):
    super().__init__(verbose=verbose, output_type=output_type)
    self.eps = eps
    self.min_samples = min_samples
    self.max_mbytes_per_batch = max_mbytes_per_batch
    self.calc_core_sample_indices = calc_core_sample_indices

# For an estimator that doesn't match any existing API:
def __init__(self, *, eps=0.5, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, verbose=False, output_type=None):
    super().__init__(verbose=verbose, output_type=output_type)
    self.eps = eps
    self.min_samples = min_samples
    self.max_mbytes_per_batch = max_mbytes_per_batch
    self.calc_core_sample_indices = calc_core_sample_indices
```

Finally, do not alter any input arguments - if you do, it will prevent proper cloning of the estimator. See Scikit-learn's [section](https://scikit-learn.org/stable/developers/develop.html#instantiation) on instantiation for more info.

For example, the following `__init__` shows what **NOT** to do:
```python
def __init__(self, my_option="option1"):
   if (my_option == "option1"):
      self.my_option = 1
   else:
      self.my_option = 2
```

This will break cloning since the value of `self.my_option` is not a valid input to `__init__`. Instead, `my_option` should be saved as an attribute as-is.

### Implementing `_get_param_names()`

To support cloning, estimators need to implement the function `_get_param_names()`. The returned value should be a list of strings of all estimator attributes that are necessary to duplicate the estimator. This method is used in `Base.get_params()` which will collect the collect the estimator param values from this list and pass this dictionary to a new estimator constructor. Therefore, all strings returned by `_get_param_names()` should be arguments in `__init__()` otherwise an invalid argument exception will be raised. Most estimators implement `_get_param_names()` similar to:

```python
@classmethod
def _get_param_names(cls):
   return super()._get_param_names() + [
      "eps",
      "min_samples",
   ]
```

**Note:** Be sure to include `super()._get_param_names()` in the returned list to properly set the `super()` attributes.

### Estimator Tags and cuML-Specific Tags

Scikit-learn introduced estimator tags in version 0.21, which are used to programmatically inspect the capabilities of estimators. These capabilities include items like sparse matrix support and the need for positive inputs, among other things. cuML estimators support _all_ of the tags defined by the Scikit-learn estimator [developer guide](https://scikit-learn.org/stable/developers/index.html), and will add support for any tag added there.

Additionally, some tags specific to cuML have been added. These tags may or may not be specific to device-accessible data types and can even apply outside of automated testing, such as allowing for the optimization of data generation. This can be useful for pipelines and HPO, among other things. These are:

- `X_types_gpu` (default=['2darray'])
   Analogous to `X_types`, indicates what types of device-accessible objects an estimator can take. `2darray` includes device-accessible ndarray objects (like CuPy and Numba) and cuDF objects, since they are all processed the same by `input_utils`. `sparse` includes `CuPy` sparse arrays.
 - `preferred_input_order` (default=None)
   One of ['F', 'C', None]. Whether an estimator "prefers" data in column-major ('F') or row-major ('C') contiguous memory layout. If different methods prefer different layouts or neither format is beneficial, then it is defined to `None` unless there is a good reason to chose either `F` or `C`. For example, all of `fit`, `predict`, etc. in an estimator use `F` but only `score` uses`C`.
- `dynamic_tags` (default=False)
   Most estimators only need to define the tags statically, which facilitates the usage of tags in general. But some estimators might need to modify the values of a tag based on runtime attributes, so this tag reflects whether an estimator needs to do that. This tag value is automatically set by the `Base` estimator class if an Estimator has defined the `_more_tags` instance method.

Note on MRO and tags: Tag resolution makes it so that multiple classes define the same tag in a composed class, classes closer to the final class overwrite the values of the farther ones. In Python, the MRO resolution makes it so that the uppermost classes are closer to the inheriting class, for example:

Class:
```python
class DBSCAN(Base,
             ClusterMixin,
             CMajorInputTagMixin):
```

MRO:
```python
>>> cuml.DBSCAN.__mro__
(<class 'cuml.cluster.dbscan.DBSCAN'>, <class 'cuml.internals.base.Base'>, <class 'cuml.internals.mixins.TagsMixin'>, <class 'cuml.internals.mixins.ClusterMixin'>, <class 'cuml.internals.mixins.CMajorInputTagMixin'>, <class 'object'>)
```

So this needs to be taken into account for tag resolution, for the case above, the tags in `ClusterMixin` would overwrite tags of `CMajorInputTagMixin` if they defined the same tags. So take this into consideration for the (uncommon) cases where there might be tags re-defined in your MRO. This is not common since most tag mixins define mutually exclusive tags (i.e. either prefer `F` or `C` major inputs).

### Estimator Array-Like Attributes

Any array-like attribute stored in an estimator needs to be convertible to the user's desired output type. To make it easier to store array-like objects in a class that derives from `Base`, the `cuml.common.array_descriptor.CumlArrayDescriptor` was created. The `CumlArrayDescriptor` class is a Python descriptor object which allows cuML to implement customized attribute lookup, storage and deletion code that can be reused on all estimators.

The `CumlArrayDescriptor` behaves different when accessed internally (from within one of `cuml`'s functions) vs. externally (for user code outside the cuml module). Internally, it behaves exactly like a normal attribute and will return the previous value set. Externally, the array will get converted to the user's desired output type lazily and repeated conversion will be cached.

Performing the array conversion lazily (i.e. converting the input array to the desired output type, only when the attribute it read from for the first time) can greatly help reduce memory consumption, but can have unintended impacts the developers should be aware of. For example, benchmarking should take into account the lazy evaluation and ensure the array conversion is included in any profiling.

#### Defining Array-Like Attributes

To use the `CumlArrayDescriptor` in an estimator, any array-like attributes need to be specified by creating a `CumlArrayDescriptor` as a class variable. An order can be specified to serve as an indicator of the order the array should be in for the C++ algorithms to work.

```python
from cuml.common.array_descriptor import CumlArrayDescriptor

class TestEstimator(cuml.Base):

   # Class variables outside of any function
   my_cuml_array_ = CumlArrayDescriptor(order='C')

   def __init__(self, ...):
      ...
```

This gives the developer full control over which attributes are arrays and the name for the array-like attribute (something that was not true before `0.17`).

#### Working with `CumlArrayDescriptor`

Once an `CumlArrayDescriptor` attribute has been defined, developers can use the attribute as they normally would. Consider the following example estimator:

```python
import cupy as cp
import cuml
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals import reflect

class SampleEstimator(cuml.Base):

   # Class variables outside of any function
   my_cuml_array_ = CumlArrayDescriptor()
   my_cupy_array_ = CumlArrayDescriptor()
   my_other_array_ = CumlArrayDescriptor()

   def __init__(self, ...):

      # Initialize to None (not mandatory)
      self.my_cuml_array_ = None

      # Init with a cupy array
      self.my_cupy_array_ = cp.zeros((10, 10))

   @reflect(reset=True)
   def fit(self, X):
      # reset=True automatically stores the type of `X` and sets n_features_in_

      # Set my_cuml_array_ with a CumlArray
      self.my_cuml_array_ = input_to_cuml_array(X, order="K").array

      # Access `my_cupy_array_` normally and set to another attribute
      # The internal type of my_other_array_ will be a CuPy array
      self.my_other_array_ = cp.ones((10, 10)) + self.my_cupy_array_

      return self
```

Just like any normal attribute, `CumlArrayDescriptor` attributes will return the same value that was set into the attribute _unless accessed externally_ (more on that below). However, developers can convert the type of an array-like attribute by using `cuml.using_output_type()` and reading from the attribute. For example, we could add a `score()` function to `TestEstimator`:

```python
def score(self):

   # Set the global output type to numpy
   with cuml.using_output_type("numpy"):
      # Accessing my_cuml_array_ will return a numpy array and
      # the result can be returned directly
      return np.sum(self.my_cuml_array_, axis=0)
```

This has the same benefits of lazy conversion and caching as when descriptors are used externally.

#### CumlArrayDescriptor External Functionality

Externally, when users read from a `CumlArrayDescriptor` attribute, the array data will be converted to the correct output type _lazily_ when the attribute is read from. For example, building off the above `TestEstimator`:

```python
my_est = SampleEstimator()

# Print the default output_type and value for `my_cuml_array_`
# By default, `output_type` is None and no global output type is set.
# Fitted descriptor attributes will mirror the fit-time input type.
print(my_est.output_type) # Output: None
print(my_est.my_cuml_array_) # Output: None
print(my_est.my_other_array_) # Output: AttributeError! my_other_array_ was never set

# Call fit() with a numpy array as the input
np_arr = np.ones((10,))
my_est.fit(np_arr) # This will load data into attributes

# `my_cuml_array_` was set internally as a CumlArray. Externally, we can check the type
print(type(my_est.my_cuml_array_)) # Output: Numpy (saved from the input of `fit`)

# Calling fit again with cupy arrays, will have a similar effect
my_est.fit(cp.ones((10,)))
print(type(my_est.my_cuml_array_)) # Output: CuPy

# Setting the `output_type` will change all descriptor properties
# and ignore the input type
my_est.output_type = "cudf"

# Reading any of the attributes will convert the type lazily
print(type(my_est.my_cuml_array_)) # Output: cuDF object

# A global output type overrides the estimator output_type attribute
with cuml.using_output_type("cupy"):
  print(type(my_est.my_cuml_array_)) # Output: cupy

# Once the global output type is restored, we return to the estimator output_type
print(type(my_est.my_cuml_array_)) # Output: cuDF. Using a cached value!
```

For more information about `CumlArrayDescriptor` and its implementation, see the [CumlArrayDescriptor Internals](#cumlarraydescriptor-internals) section of the Appendix.

### Estimator Methods

cuML uses the reflection system to manage output type conversions, ensuring arrays are returned in the user's expected format (cupy, numpy, pandas, cudf, etc.). The `@reflect` decorator and related utilities handle this automatically.

For comprehensive documentation, see the [Reflection Guide](REFLECTION_GUIDE.md).

#### Using the `@reflect` Decorator

The `@reflect` decorator should be used on methods that return arrays to the user:

```python
import cupy as cp
import cuml
from cuml.common import input_to_cuml_array
from cuml.internals import reflect

class MyEstimator(cuml.Base):
    @reflect(reset=True)
    def fit(self, X, y=None):
        self.coef_ = input_to_cuml_array(X, order="K").array
        return self

    @reflect
    def predict(self, X):
        X_m = input_to_cuml_array(X, order="K").array
        result = cp.asarray(X_m) + cp.ones(X_m.shape)
        return result  # Can return any array-like object
```

| Decorator Usage | When to Use |
| :-------------- | :---------- |
| `@reflect(reset=True)` | Fit-like methods where the decorator should store `_input_type` and set `n_features_in_`. |
| `@reflect(reset="type")` | Fit-like methods that store `_input_type` through reflection but set or check `n_features_in_` through custom validation. |
| `@reflect` | Transform/predict methods that return arrays. |
| `@reflect(array=None)` | Methods with no array input (e.g., `forecast(nsteps)`). Uses fit-time input type. |

#### Handling Special Cases

For methods that need manual output handling (e.g., classifier `predict` with label decoding), use `@run_in_internal_context` with `exit_internal_context`:

```python
import cupy as cp
import cuml
from cuml.internals import run_in_internal_context, exit_internal_context

class MyClassifier(cuml.Base):
    @run_in_internal_context
    def predict(self, X):
        # Call reflected method - returns CumlArray internally
        scores = self.decision_function(X)

        # Manual processing
        indices = (scores.to_output("cupy") >= 0).view(cp.int8)

        # Exit internal context to get proper output type
        with exit_internal_context():
            output_type = self._get_output_type(X)
        return decode_labels(indices, self.classes_, output_type=output_type)
```

#### Score Methods

Score methods typically return scalars, not arrays. Use `@run_in_internal_context`:

```python
@run_in_internal_context
def score(self, X, y):
    predictions = self.predict(X)
    return accuracy_score(y, predictions)
```

#### Property Accessors

For properties that return arrays:

```python
@property
@reflect
def support_(self):
    return self._support_vectors
```

## Do's And Do Not's

### **Do:** Use the `@reflect` Decorator on Public API Methods

Use `@reflect` on methods that return arrays to users. Use `@reflect(reset=True)` for simple fit-like methods, and `@reflect(reset="type")` when a fit-like method performs its own feature validation.

**Do this:**
```python
@reflect(reset=True)
def fit(self, X, y, convert_dtype=True) -> "KNeighborsRegressor":
    ...

@reflect
def predict(self, X, convert_dtype=True) -> CumlArray:
    ...

@reflect
def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity') -> SparseCumlArray:
    ...
```

**Not this (missing decorators):**
```python
def fit(self, X, y, convert_dtype=True):
    ...

def predict(self, X, convert_dtype=True):
    ...
```

### **Do:** Return Array-Like Objects Directly

There is no need to convert the array type before returning it. Simply return any array-like object and the `@reflect` decorator will automatically convert it.

**Do this:**
```python
@reflect
def predict(self) -> CumlArray:
   cp_arr = cp.ones((10,))

   return cp_arr
```

**Not this:**
```python
@reflect
def predict(self, X, y) -> CumlArray:
   cp_arr = cp.ones((10,))

   # Don't be tempted to use `CumlArray(cp_arr)` here either
   cuml_arr = input_to_cuml_array(cp_arr, order="K").array

   return cuml_arr.to_output(self._get_output_type(X))
```

### **Don't:** Use `CumlArray.to_output()` directly

Using `CumlArray.to_output()` is no longer necessary except in very rare circumstances. The `@reflect` decorator handles output type conversion automatically. For internal conversions, use `cuml.using_output_type()` context manager.

**Do this:**
```python
@reflect
def _private_func(self) -> CumlArray:
   return cp.ones((10,))

@reflect
def predict(self, X, y) -> CumlArray:
   self.my_cupy_attribute_ = cp.zeros((10,))

   with cuml.using_output_type("numpy"):
      np_arr = self._private_func()

   return self.my_cupy_attribute_ + np_arr
```

**Not this:**
```python
def _private_func(self) -> CumlArray:
   return cp.ones((10,))

def predict(self, X, y) -> CumlArray:
   self.my_cupy_attribute_ = cp.zeros((10,))

   np_arr = CumlArray(self._private_func()).to_output("numpy")

   return CumlArray(self.my_cupy_attribute_).to_output("numpy") + np_arr
```

### **Don't:** Perform parameter modification in `__init__()`

Input arguments to `__init__()` should be stored as they were passed in. Parameter modification, such as converting parameter strings to integers, should be done in `fit()` or a helper private function.

While it's more verbose, altering the parameters in `__init__` will break the estimator's ability to be used in `clone()`.

**Do this:**
```python
class TestEstimator(cuml.Base):

   def __init__(self, method_name: str, ...):
      super().__init__(...)

      self.method_name = method_name

   def _method_int(self) -> int:
      return 1 if self.method_name == "type1" else 0

   @reflect(reset=True)
   def fit(self, X) -> "TestEstimator":

      # Call external code from Cython
      my_external_func(X.ptr, <int>self._method_int())

      return self
```

**Not this:**
```python
class TestEstimator(cuml.Base):

   def __init__(self, method_name: str, ...):
      super().__init__(...)

      self.method_name = 1 if method_name == "type1" else 0

   @reflect(reset=True)
   def fit(self, X) -> "TestEstimator":

      # Call external code from Cython
      my_external_func(X.ptr, <int>self.method_name)

      return self
```

## Appendix

This section contains more in-depth information about the descriptors and internal mechanisms to help developers understand what's going on behind the scenes.

### Reflection System

For detailed documentation on the reflection system (`@reflect`, `@run_in_internal_context`, `exit_internal_context`), see the [Reflection Guide](REFLECTION_GUIDE.md).

### Estimator Array-Like Attributes

#### `CumlArrayDescriptor` Internals

The internal representation of `CumlArrayDescriptor` is a `CumlArrayDescriptorMeta` object. To inspect the internal representation, the attribute value must be directly accessed from the estimator's `__dict__` (`getattr` and `__getattr__` will perform the conversion). For example:

```python
my_est = TestEstimator()
my_est.fit(cp.ones((10,)))

# Access the CumlArrayDescriptorMeta value directly. No array conversion will occur
print(my_est.__dict__["my_cuml_array_"])
# Output: CumlArrayDescriptorMeta(input_type='cupy', values={'cuml': <cuml.internals.array.CumlArray object at 0x7fd39174ae20>, 'numpy': array([ 0,  1,  1,  2,  2, -1, -1, ...

# Values from CumlArrayDescriptorMeta can be specifically read
print(my_est.__dict__["my_cuml_array_"].input_type)
# Output: "cupy"

# The input value can be accessed
print(my_est.__dict__["my_cuml_array_"].get_input_value())
# Output: CumlArray ...
```
