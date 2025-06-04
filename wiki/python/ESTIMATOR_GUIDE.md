# cuML Python Estimators Developer Guide

This guide is meant to help developers follow the correct patterns when creating/modifying any cuML Estimator object and ensure a uniform cuML API.

**Note:** This guide is long, because it includes internal details on how cuML manages input and output types for advanced use cases. But for the vast majority of estimators, the requirements are very simple and can follow the example patterns shown below in the [Quick Start Guide](#quick-start-guide).

## Table of Contents

- [Recommended Scikit-Learn Documentation](#recommended-scikit-learn-documentation)
- [API Matching Policy](#api-matching-policy)
- [Quick Start Guide](#quick-start-guide)
- [Background](#background)
   - [Input and Output Types in cuML](#input-and-output-types-in-cuml)
   - [Specifying the Array Output Type](#specifying-the-array-output-type)
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
1. Inherit from `cuml.common.base.Base`
   ```python
   from cuml.common.base import Base

   class MyEstimator(Base):
      ...
   ```
2. Follow the Scikit-learn estimator guidelines found [here](https://scikit-learn.org/stable/developers/develop.html)
3. Include the `Base.__init__()` arguments available in the new Estimator's `__init__()`
   ```python
   class MyEstimator(Base):

      def __init__(self, *, extra_arg=True, handle=None, verbose=logger.level_enum.info, output_type=None):
         super().__init__(handle=handle, verbose=verbose, output_type=output_type)
         ...
   ```
4. Declare each array-like attribute the new Estimator will compute as a class variable for automatic array type conversion. An order can be specified to serve as an indicator of the order the array should be in for the C++ algorithms to work.
   ```python
   from cuml.common.array_descriptor import CumlArrayDescriptor

   class MyEstimator(Base):

      labels_ = CumlArrayDescriptor(order='C')

      def __init__(self):
         ...
   ```
5. Add input and return type annotations to public API functions OR wrap those functions explicitly with conversion decorators (see [this example](#non-standard-predict) for a non-standard use case)
   ```python
   class MyEstimator(Base):

      def fit(self, X) -> "MyEstimator":
         ...

      def predict(self, X) -> CumlArray:
         ...
   ```
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

## Background

Some background is necessary to understand the design of estimators and how to work around any non-standard situations.

### Input and Output Types in cuML

In cuML we support both ingesting and generating a variety of different object types. Estimators should be able to accept and return any array type. The types that are supported as of release 0.17:

 - cuDF DataFrame or Series
 - Pandas DataFrame or Series
 - NumPy Arrays
 - Numba Device Arrays
 - CuPy arrays
 - CumlArray type (Internal to the `cuml` API only.)

When converting between types, it's important to minimize memory accessibility transitions as much as possible. Conversions such as NumPy -> CuPy or Numba -> Pandas DataFrame will incur a performance penalty as memory is copied between different accessibility domains.

Converting between types within the same accessibility domain (e.g., NumPy -> Pandas or CuPy -> cuDF) do not have as significant of a penalty, though they may still increase memory usage (this is particularly true for the array <-> dataframe conversion. i.e. when converting from CuPy to cuDF, memory usage may increase slightly).

Finally, conversions between Numba<->CuPy<->CumlArray incur the least amount of overhead since only the device pointer is moved from one class to another.

Internally, all arrays should be converted to `CumlArray` as much as possible since it is compatible with all output types and can be easily converted.

### Host and Device Arrays

cuML provides flexible memory management through the `CumlArray` class, which can store data in either device-accessible or host-accessible memory. This flexibility allows for efficient data handling and computation.

To control where arrays are stored, the `memory_type` setting determines where arrays are stored by default. This setting can be configured through:

1. Global settings:
   ```python
   import cuml
   cuml.global_settings.memory_type = 'device'  # For device-accessible memory
   ```

2. Context managers:
   ```python
   with cuml.using_memory_type('device'):
       # Arrays created here will use device-accessible memory
       pass
   ```

3. Individual estimator settings:
   ```python
   estimator = MyEstimator(output_mem_type='device')
   ```

The memory type settings allow for efficient data management:
- Use device-accessible memory for GPU-accelerated computations
- Use host-accessible memory for data that needs to be accessed by the CPU
- Let cuML handle memory transfers automatically when needed

New array output types were introduced to take advantage of these settings by deferring to the globally-set memory type. Read on for more details on how to take advantage of these types.

### Specifying the Array Output Type

Users can choose which array type should be returned by cuml by either:
1. Individually setting the output_type property on an estimator class (i.e `Base(output_type="numpy")`)
2. Globally setting the `cuml.global_output_type`
3. Temporarily setting the `cuml.global_output_type` via the `cuml.using_output_type` context manager

**Note:** Setting `cuml.global_output_type` (either directly or via `cuml.set_output_type()` or `cuml.using_output_type()`) <u>will take precedence over any value in `Base.output_type</u>

In addition, for developers, it is sometimes useful to set the output memory
type separately from the output type, as will be described in further
detail below. **End-users will not typically use this setting
themselves.** To set the output memory type, developers can:
1. Individually setting the output_mem_type property on an estimator class that derives from `UniversalBase` (i.e `UniversalBase(output_mem_type="host")`)
2. Globally setting the `cuml.global_settings.memory_type`
3. Temporarily setting the `cuml.global_settings.memory_type` via the `cuml.using_memory_type` context manager

Changing the array output type will alter the return value of estimator functions (i.e. `predict()`, `transform()`), and the return value for array-like estimator attributes (i.e. `my_estimator.classes_` or `my_estimator.coef_`)

All output_types (including `cuml.global_output_type`) are specified using an all lowercase string. These strings can be passed in an estimators constructor or via `cuml.set_global_output_type` and `cuml.using_output_type`. Accepted values are:

 - `None`: (Default) No global value set. Will use individual values from estimators output_type
 - `"input"`: Similar to `None`. Will mirror the same type as any array passed into the estimator
 - `"array"`: Returns Numpy or Cupy arrays depending on the current memory
   type
 - `"numba"`: Returns Numba Device Arrays
 - `"dataframe"`: Returns cuDF or Pandas DataFrames depending on the
   current memory type
 - `"series"`: Returns cuDF or Pandas Series depending on the current memory
   type
 - `"df_obj"`: Returns cuDF/Pandas Series if array is single-dimensional or
   cuDF/Pandas DataFrames otherwise
 - `"cupy"`: Returns CuPy Device Arrays
 - `"numpy"`: Returns Numpy Arrays
 - `"cudf"`: Returns cuDF DataFrame if cols > 1, else cuDF Series
 - `"pandas"`: Returns Pandas DataFrame if cols > 1, else cuDF Series

**Note:** There is an additional option `"mirror"` which can only be set by internal API calls and is not user accessible. This value is only used internally by the `CumlArrayDescriptor` to mirror any input value set.

#### Deferring to the global memory type

When working with array types, it's often useful to request generic types like "array" or "dataframe" rather than specifying concrete types like cupy/numpy or cudf/pandas. This allows the code to adapt to the current memory accessibility settings automatically.

For example, if a developer needs to use methods specific to array types (like cupy/numpy) that aren't available on the generic `CumlArray` interface, they can use these generic output types to ensure compatibility with the current memory accessibility settings:

```python
# Instead of hardcoding a specific type
arr.to_output('cupy')  # May not be appropriate for all memory settings

# Use a generic type that adapts to current settings
arr.to_output('array')  # Will use appropriate type based on memory settings
```

The following generic output types are available:
- `array`: Returns arrays appropriate for the current memory settings
- `series`: Returns series appropriate for the current memory settings
- `dataframe`: Returns dataframes appropriate for the current memory settings
- `df_obj`: Returns series for single-dimensional data or dataframes for multi-dimensional data

It's recommended to use these generic output types for internal conversion calls to ensure compatibility with different memory accessibility configurations. For cases where a specific memory type is required, it should be specified directly to maintain clarity about the memory requirements.

External users typically won't need to use these generic types unless they're specifically working with applications that need to adapt to different memory accessibility configurations.

### Ingesting Arrays

When the input array type isn't known, the correct and safest way to ingest arrays is using `cuml.common.input_to_cuml_array`. This method can handle all supported types, is capable of checking the array order, can enforce a specific dtype, and can raise errors on incorrect array sizes:

```python
def fit(self, X):
    cuml_array, dtype, cols, rows = input_to_cuml_array(X, order="K")
    ...
```

### Returning Arrays

The `CumlArray` class can convert to any supported array type using the `to_output(output_type: str)` method. However, doing this explicitly is almost never needed in practice and **should be avoided**. Directly converting arrays with `to_output()` will circumvent the automatic conversion system potentially causing extra or incorrect array conversions.

## Estimator Design

All estimators (any class that is a child of `cuml.common.base.Base`) have a similar structure. In addition to the guidelines specified in the [SkLearn Estimator Docs](https://scikit-learn.org/stable/developers/develop.html), cuML implements a few additional rules.

### Initialization

All estimators should match the arguments (including the default value) in `Base.__init__` and pass these values to `super().__init__()`. As of 0.17, all estimators should accept `handle`, `verbose` and `output_type`.

In general, all estimator constructor parameters should be keyword-only except for those arguments that are not keyword-only in the matched API. This helps prevent breaking changes if arguments are added or removed in future versions. For example:

```python
# For an estimator that matches scikit-learn's API where the eps argument can be positional:
def __init__(self, eps=0.5, *, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, handle=None, verbose=False, output_type=None):
    super().__init__(handle=handle, verbose=verbose, output_type=output_type)
    self.eps = eps
    self.min_samples = min_samples
    self.max_mbytes_per_batch = max_mbytes_per_batch
    self.calc_core_sample_indices = calc_core_sample_indices

# For an estimator that doesn't match any existing API:
def __init__(self, *, eps=0.5, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, handle=None, verbose=False, output_type=None):
    super().__init__(handle=handle, verbose=verbose, output_type=output_type)
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

Note on MRO and tags: Tag resolution makes it so that multiple classes define the same tag in a composed class, classes closer to the final class overwrite the values of the farther ones. In Python, the MRO resolution makes it so that the uppermost classes are closer to the inheritting class, for example:

Class:
```python
class DBSCAN(Base,
             ClusterMixin,
             CMajorInputTagMixin):
```

MRO:
```python
>>> cuml.DBSCAN.__mro__
(<class 'cuml.cluster.dbscan.DBSCAN'>, <class 'cuml.common.base.Base'>, <class 'cuml.common.mixins.TagsMixin'>, <class 'cuml.common.mixins.ClusterMixin'>, <class 'cuml.common.mixins.CMajorInputTagMixin'>, <class 'object'>)
```

So this needs to be taken into account for tag resolution, for the case above, the tags in `ClusterMixin` would overwrite tags of `CMajorInputTagMixin` if they defined the same tags. So take this into consideration for the (uncommon) cases where there might be tags re-defined in your MRO. This is not common since most tag mixins define mutually exclusive tags (i.e. either prefer `F` or `C` major inputs).

### Estimator Array-Like Attributes

Any array-like attribute stored in an estimator needs to be convertible to the user's desired output type. To make it easier to store array-like objects in a class that derives from `Base`, the `cuml.common.array.CumlArrayDescriptor` was created. The `CumlArrayDescriptor` class is a Python descriptor object which allows cuML to implement customized attribute lookup, storage and deletion code that can be reused on all estimators.

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

   def fit(self, X):
      # Stores the type of `X` and sets the output type if self.output_type == "input"
      self._set_output_type(X)

      # Set my_cuml_array_ with a CumlArray
      self.my_cuml_array_, *_ = input_to_cuml_array(X, order="K")

      # Access `my_cupy_array_` normally and set to another attribute
      # The internal type of my_other_array_ will be a CuPy array
      self.my_other_array_ = cp.ones((10, 10)) + self.my_cupy_array_

      return self
```

Just like any normal attribute, `CumlArrayDescriptor` attributes will return the same value that was set into the attribute _unless accessed externally_ (more on that below). However, developers can convert the type of an array-like attribute by using the `cuml.global_output_type` functionality and reading from the attribute. For example, we could add a `score()` function to `TestEstimator`:

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
# By default, `output_type` is set to `cuml.global_output_type`
# If `cuml.global_output_type == None`, `output_type` is set to "input"
print(my_est.output_type) # Output: "input"
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

# Setting the global_output_type, overrides the estimator output_type attribute
with cuml.using_output_type("cupy"):
  print(type(my_est.my_cuml_array_)) # Output: cupy

# Once the global_output_type is restored, we return to the estimator output_type
print(type(my_est.my_cuml_array_)) # Output: cuDF. Using a cached value!
```

For more information about `CumlArrayDescriptor` and it's implementation, see the [CumlArrayDescriptor Details]() section of the Appendix.

### Estimator Methods

To allow estimator methods to accept a wide variety of inputs and outputs, a set of decorators have been created to wrap estimator functions (and all `cuml` API functions as well) and perform the standard conversions automatically. cuML provides 2 options to for performing the standard array type conversions:

1. For many common patterns used in functions like `fit()`, `predict()`, `transform()`, `cuml.Base` can automatically perform the data conversions as long as a method has the necessary type annotations.
2. Decorators can be manually added to methods to handle more advanced use cases

#### Option 1: Automatic Array Conversion From Type Annotation

To automatically convert array-like objects being returned by an Estimator method, a new metaclass has been added to `Base` that can scan the return type information of an Estimator method and infer which, if any, array conversion should be done. For example, if a method returns a type of `Base`, cuML can assume this method is likely similar to `fit()` and should call `Base._set_base_attributes()` before calling the method. If a method returns a type of `CumlArray`, cuML can assume this method is similar to `predict()` or `transform()`, and the return value is an array that may need to be converted using the output type calculated in `Base._get_output_type()`.

The full set of return types rules that will be applied by the `Base` metaclass are:

| Return Type | Converts Array Type? | Common Methods | Notes |
| :---------: | :-----------: | :----------- | :----------- |
| `Base` | No | `fit()` | Any type that inherits or `isinstance` of `Base` will work |
| `CumlArray` | Yes | `predict()`, `transform()` | Functions can return any array-like object (`np.ndarray`, `cp.ndarray`, etc. all accepted) |
| `SparseCumlArray` | Yes | `predict()`, `transform()` | Functions can return any sparse array-like object (`scipy`, `cupyx.scipy` sparse arrays accepted) |
| `dict`, `tuple`, `list` or `typing.Union` | Yes |  | Functions must return a generic object that contains an array-like object. No sparse arrays are supported |

Simply setting the return type of a method is all that is necessary to automatically convert the return type (with the added benefit of adding more information to the code). Below are some examples to show simple methods using automatic array conversion.

##### `fit()`

```python
def fit(self, X) -> "KMeans":

   # Convert the input to CumlArray
   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

**Notes:**
 - Any type that derives from `Base` can be used as the return type for `fit()`. In python, to indicate returning `self` from a function, class type can be surrounded in quotes to prevent an import error.


##### `predict()`

```python
def predict(self, X) -> CumlArray:
   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)

   # Directly return a cupy array
   return X_m
```

**Notes:**
 - It's not necessary to convert to `CumlArray` and cast with `to_output` before returning. This function directly returned a `cp.ndarray` object. Any array-like object can be returned.

#### Option 2: Manual Estimator Method Decoration

While the automatic conversions from type annotations works for many estimator functions, sometimes its necessary to explicitly decorate an estimator method. This allows developers greater flexibility over the input argument, output type and output dtype.

Which decorator to use for an estimator function is determined by 2 factors:

1. Function return type
2. Whether the function is on a class deriving from `Base`

The full set of descriptors can be organized by these two factors:

| Return Type-> | Array-Like | Sparse Array-Like | Generic | Any |
| -----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `Base`      | `@api_base_return_array` | `@api_base_return_sparse_array` |`@api_base_return_generic` | `@api_base_return_any` |
| Non-`Base`   | `@api_return_array` | `@api_return_sparse_array` | `@api_return_generic` | `@api_return_any` |

Simply choosing the decorator based off the return type and if the function is on `Base` will work most of the time. The decorator default options were designed to work on most estimator functions without much customization.

An in-depth discussion of how these decorators work, when each should be used, and their default options can be found in the Appendix. For now, we will show an example method that uses a non-standard input argument name, and also requires converting the array dtype:

##### Non-Standard `predict()`

```python
@cuml.internals.api_base_return_array(input_arg="X_in", get_output_dtype=True)
def predict(self, X):
   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)

   # Return the cupy array directly
   return X_m
```

**Notes:**
 - The decorator argument `input_arg` can be used to specify which input should be considered the "input".
   - In reality, this isn't necessary for this example. The decorator will look for an argument named `"X"` or default to the first, non `self`, argument.
 - It's not necessary to convert to `CumlArray` and casting with `to_output` before returning. This function directly returned a `cp.ndarray` object. Any array-like object can be returned.
 - Specifying `get_output_dtype=True` in the decorator argument instructs the decorator to also calculate the dtype in addition to the output type.

## Do's And Do Not's

### **Do:** Add Return Typing Information to Estimator Functions

Adding the return type to estimator functions will allow the `Base` meta-class to automatically decorate functions based on their return type.

**Do this:**
```python
def fit(self, X, y, convert_dtype=True) -> "KNeighborsRegressor":
def predict(self, X, convert_dtype=True) -> CumlArray:
def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity') -> SparseCumlArray:
def predict(self, start=0, end=None, level=None) -> typing.Union[CumlArray, float]:
```

**Not this:**
```python
def fit(self, X, y, convert_dtype=True):
def predict(self, X, convert_dtype=True):
def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
def predict(self, start=0, end=None, level=None):
```

### **Do:** Return Array-Like Objects Directly

There is no need to convert the array type before returning it. Simply return any array-like object and it will be automatically converted

**Do this:**
```python
def predict(self) -> CumlArray:
   cp_arr = cp.ones((10,))

   return cp_arr
```

**Not this:**
```python
def predict(self, X, y) -> CumlArray:
   cp_arr = cp.ones((10,))

   # Don't be tempted to use `CumlArray(cp_arr)` here either
   cuml_arr = input_to_cuml_array(cp_arr, order="K").array

   return cuml_arr.to_output(self._get_output_type(X))
```

### **Don't:** Use `CumlArray.to_output()` directly

Using `CumlArray.to_output()` is no longer necessary except in very rare circumstances. Converting array types is best handled with `input_to_cuml_array` or `cuml.using_output_type()` when retrieving `CumlArrayDescriptor` values.

**Do this:**
```python
def _private_func(self) -> CumlArray:
   return cp.ones((10,))

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

   def fit(self, X) -> "TestEstimator":

      # Call external code from Cython
      my_external_func(X.ptr, <int>self.method_name)

      return self
```

## Appendix

This section contains more in-depth information about the decorators and descriptors to help developers understand whats going on behind the scenes

### Estimator Array-Like Attributes

#### Automatic Decoration Rules

Adding decorators to every estimator function just to use the decorator default values would be very repetitive and unnecessary. Because most of estimator functions follow a similar pattern, a new meta-class has been created to automatically decorate estimator functions based off their return type. This meta class will decorate functions according to a few rules:

1. If a functions has been manually decorated, it will not be automatically decorated
2. If an estimator function returns an instance of `Base`, then `@api_base_return_any()` will be applied.
3. If an estimator function returns a `CumlArray`, then `@api_base_return_array()` will be applied.
3. If an estimator function returns a `SparseCumlArray`, then `@api_base_return_sparse_array()` will be applied.
4. If an estimator function returns a `dict`, `tuple`, `list` or `typing.Union`, then `@api_base_return_generic()` will be applied.

| Return Type | Decorator | Notes |
| :-----------: | :-----------: | :----------- |
| `Base` | `@api_base_return_any(set_output_type=True, set_n_features_in=True)` | Any type that `isinstance` of `Base` will work |
| `CumlArray` | `@api_base_return_array(get_output_type=True)` | Functions can return any array-like object |
| `SparseCumlArray` | `@api_base_return_sparse_array(get_output_type=True)` | Functions can return any sparse array-like object |
| `dict`, `tuple`, `list` or `typing.Union` | `@api_base_return_generic(get_output_type=True)` | Functions must return a generic object that contains an array-like object. No sparse arrays are supported |

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

### Estimator Methods

#### Common Functionality

All of these decorators perform the same basic steps with a few small differences. The common steps performed by each decorator is:

1. Set `cuml.global_output_type = "mirror"`
   1. When `"mirror"` is used as the global output type, that indicates we are in an internal cuML API call. The `CumlArrayDescriptor` keys off this value to change between internal and external functionality
2. Set CuPy allocator to use RMM
   1. This replaces the existing decorator `@with_cupy_rmm`
   2. Unlike before, the CuPy allocator is only set once per API call
3. Set the estimator input attributes. Can be broken down into 3 steps:
   1. Set `_input_type` attribute
   2. Set `target_dtype` attribute
   3. Set `n_features` attribute
4. Call the desired function
5. Get the estimator output type. Can be broken down into 2 steps:
   1. Get `output_type`
   2. Get `output_dtype`
6. Convert the return value
   1. This will ultimately call `CumlArray.to_output(output_type=output_type, output_dtype=output_dtype)

While the above list of steps may seem excessive for every call, most functions follow this general form, but may skip a few steps depending on a couple of factors. For example, Step #3 is necessary on functions that modify the estimator's estimated attributes, such as `fit()`, but is not necessary for functions like `predict()` or `transform()`. And Step #5/6 are only necessary when returning array-like objects and are omitted when returning any other type.

Functionally, you can think of these decorators equivalent to the following pseudocode:
```python
def my_func(self, X):
   with cuml.using_ouput_type("mirror"):
      with cupy.cuda.cupy_using_allocator(
          rmm.allocators.cupy.rmm_cupy_allocator
      ):
         # Set the input properties
         self._set_base_attributes(output_type=X, n_features=X)

         # Do actual calculation returning an array-like object
         ret_val = self._my_func(X)

         # Get the output type
         output_type = self._get_output_type(X)

         # Convert array-like to CumlArray
         ret_val = input_to_cuml_array(ret_val, order="K").array

         # Convert CumlArray to desired output_type
         return ret_val.to_output(output_type)
```

Keep the above pseudocode in mind when working with these decorators since their goal is to replace many of these repetitive functions.

### Decorator Defaults

Every function in `cuml` is slightly different and some `fit()` functions may need to set the `target_dtype` or some `predict()` functions may need to skip getting the output type. To handle these situations, all of the decorators take arguments to configure their functionality.

Since the decorator's functionality is very similar, so are their arguments. All of the decorators take similar arguments that will be outlined below.

| Argument | Type | Default | Meaning |
| :-----------: | :-----------: | :-----------: | :----------- |
| `input_arg` | `str` | `'X'` or 1st non-self argument | Determines which input argument to use for `_set_output_type()` and `_set_n_features_in()` |
| `target_arg` | `str` | `'y'` or 2nd non-self argument | Determines which input argument to use for `_set_target_dtype()` |
| `set_output_type` | `bool` | Varies | Whether to call `_set_output_type(input_arg)` |
| `set_output_dtype` | `bool` | `False` | Whether to call `_set_target_dtype(target_arg)` |
| `set_n_features_in` | `bool` | Varies | Whether to call `_set_n_features_in(input_arg)` |
| `get_output_type` | `bool` | Varies | Whether to call `_get_output_type(input_arg)` |
| `get_output_dtype` | `bool` | `False` | Whether to call `_get_target_dtype()` |

An example of how these arguments can be used is below:

**Before:**
```python
@with_cupy_rmm
def predict(self, X, y):
   # Determine the output type and dtype
   out_type = self._get_output_type(y)
   out_dtype = self._get_target_dtype()

   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Convert the CudaArray to the desired output
   return X_m.to_output(output_type=out_type, output_dtype=out_dtype)
```

**After:**
```python
@cuml.internals.api_base_return_array(input_arg="y", get_output_dtype=True)
def predict(self, X):
   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Convert the CudaArray to the desired output
   return X_m
```

#### Before `0.17` and After Comparison

For developers used to the `0.16` architecture it can be helpful to see examples of estimator methods from `0.16` compared to `0.17` and after. This section shows a few examples side-by-side to illustrate the changes.

##### `fit()`

<table>
   <thead>
      <tr>
         <th>Before</th>
         <th>After</th>
      </tr>
   </thead>
<tbody>
   <tr>
   <td>

```python
@with_cupy_rmm
def fit(self, X):
   # Set the base input attributes
   self._set_base_attributes(output_type=X, n_features=X)

   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

   </td>
   <td stype="text-align: top">

```python

def fit(self, X) -> "KMeans":



   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

   </td>
   </tr>
</tbody>
</table>

**Notes:**
 - `@with_cupy_rmm` is no longer needed. This is automatically applied for every public method of estimators
 - `self._set_base_attributes()` no longer needs to be called.

##### `predict()`

<table>
   <thead>
      <tr>
         <th>Before</th>
         <th>After</th>
      </tr>
   </thead>
<tbody>
   <tr>
   <td>

```python
@with_cupy_rmm
def predict(self, X, y):
   # Determine the output type and dtype
   out_type = self._get_output_type(y)

   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Do some calculation with cupy
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)

   # Convert back to CumlArray
   X_m = CumlArray(X_m)

   # Convert the CudaArray to the desired output
   return X_m.to_output(output_type=out_type)
```
   </td>
   <td stype="text-align: top">

```python

def predict(self, X) -> CumlArray:



   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)




   # Directly return a cupy array
   return X_m
```

   </td>
   </tr>
</tbody>
</table>

**Notes:**
 - `@with_cupy_rmm` is no longer needed. This is automatically applied for every public method of estimators
 - `self._get_output_type()` no longer needs to be called. The output type is determined automatically
 - Its not necessary to convert to `CumlArray` and casting with `to_output` before returning. This function directly returned a `cp.ndarray` object. Any array-like object can be returned.

##### `predict()` with `dtype`

<table>
   <thead>
      <tr>
         <th>Before</th>
         <th>After</th>
      </tr>
   </thead>
<tbody>
   <tr>
   <td>

```python

@with_cupy_rmm
def predict(self, X_in):
   # Determine the output_type
   out_type = self._get_output_type(X_in)
   out_dtype = self._get_target_dtype()

   # Convert to CumlArray
   X_m = input_to_cuml_array(X_in, order="K").array

   # Call a cuda function
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)

   # Convert back to CumlArray
   X_m = CumlArray(X_m)

   # Convert the CudaArray to the desired output and dtype
   return X_m.to_output(output_type=out_type, output_dtype=out_dtype)
```

   </td>
   <td stype="text-align: top">

```python

@api_base_return_array(input_arg="X_in", get_output_dtype=True)
def predict(self, X):




   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   X_m = cp.asarray(X_m) + cp.ones(X_m.shape)




   # Return the cupy array directly
   return X_m

```

   </td>
   </tr>
</tbody>
</table>

**Notes:**
 - `@with_cupy_rmm` is no longer needed. This is automatically applied with every decorator
 - The decorator argument `input_arg` can be used to specify which input should be considered the "input".
   - In reality, this isn't necessary for this example. The decorator will look for an argument named `"X"` or default to the first, non `self`, argument.
 - `self._get_output_type()` and `self._get_target_dtype()` no longer needs to be called. Both the output type and dtype are determined automatically
 - It's not necessary to convert to `CumlArray` and casting with `to_output` before returning. This function directly returned a `cp.ndarray` object. Any array-like object can be returned.
 - Specifying `get_output_dtype=True` in the decorator argument instructs the decorator to also calculate the dtype in addition to the output type.
