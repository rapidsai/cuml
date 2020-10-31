# cuML Python Estimators Developer Guide

This guide is meant to help developers follow the correct patterns when creating/modifying any class that inherits from `cuml.common.base.Base`.

Start by reading the following:

1. [SkLearn Estimator Docs](https://scikit-learn.org/stable/developers/develop.html)
   1. `cuml` Estimator design follows Sklearn very closely. We will only cover portions where our design differs from this document
   2. Pay close attention to these sections as these topics have caused pain points in the past:
      1. [Instantiation](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      2. [Estimated Attributes](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      3. [`get_params` and `set_params`](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      4. [cloning](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
2. [SkLearn Docstring Guide](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation)
   1. We follow the same guidlines for speficying array-like objects, array shapes, dtypes, and default values

## Background

Some background is necessary to understand the design of estimators and how to work around any non-standard situations.

### Input and Output Types in cuML

In `cuml` we support both ingesting and generating a variety of different n-dimensional array types. Estimators should be able to accept and return any array type. The types that are supported as of release 0.17:

 - cuDF DataFrame or Series
 - Pandas DataFrame or Series
 - NumPy Arrays
 - Numba Device Arrays
 - CuPy arrays
 - CumlArray type (Internal to the `cuml` API only.)

When converting between types, it's important to minimize the CPU<->GPU type conversions as much as possible. Conversions such as NumPy -> CuPy or Numba -> Pandas DataFrame will incur a performance penalty as memory is copied from device to host or viceversa.

Converting between types of the same device, i.e. CPU<->CPU or GPU<->, do not have as significant of a penalty.

Finally, conversions between Numba<->CuPy<->CumlArray incur the least amount of overhead since only the device pointer is moved from one class to another.

Internally, all array's should be converted to `CumlArray` as much as possible since it is compatible with all output types and can be easily converted.

### Speficying the Array Output Type

Users can choose which array type should be returned by cuml by either: 
1. Individually setting the output_type property on an estimator class (i.e `Base(output_type="numpy")`)
2. Globally setting the `cuml.global_output_type`

Note: Setting the global output type will take precedence over any value in `Base.output_type`

Changing the array output type will alter the return value of estimator functions (i.e. `predict()`, `transform()`), and the return value for estimator array-like types (i.e. `my_estimator.classes_`)

All output_types (including `cuml.global_output_type`) are specfied using an all lowercase string. These strings can be passed in an estimators constructor or via `cuml.set_global_output_type` and `cuml.using_output_type`. Accepted values are:

 - `None`: (Default) No global value set. Will use individual values from estimators output_type
 - `"input"`: Similar to `None`. Will mirror the same type as any array passed into the estimator
 - `"mirror"`: Special value for handling internal `cuml` API calls. Will mirror the input value set directly
   - Note: Cannot be directly set by users
 - `"numba"`: Returns Numba Device Arrays
 - `"numpy"`: Returns Numpy Arrays
 - `"cudf"`: Returns cuDF DataFrame if cols > 1, else cuDF Series
 - `"cupy"`: Returns CuPy Device Arrays

### Ingesting Arrays

When the input array type isn't know, the correct and safest way to ingest arrays is using `cuml.common.input_to_cuml_array`. This method can handle all supported types, is capable of checking the array order, can enforce a specific dtype, and can raise errors on incorrect array sizes:

```python
cuml_array, dtype, cols, rows = input_to_cuml_array(X, order="K")
```

### Returning Arrays

The `CumlArray` class can convert to any supported array type using the `to_output(output_type: str)` method. However, this is almost never needed in practice and **should be avoided**.

## Estimator Design

All estimators (any class that is a child of `cuml.common.base.Base`) have a similar structure. In addition to the guidelines specified in the [SkLearn Estimator Docs](https://scikit-learn.org/stable/developers/develop.html), cuML implements a few additional rules. It's suggested to read the SkLearn Estimator Doc before continuing.

### Arguments to `__init__`

All estimators should match the arguments (including the default value) in `Base.__init__` and pass these values to `super().__init__()`. As of 0.17, all estimators should accept `handle`, `verbose` and `output_type`.

In addition, is recommended to force keyword arguments to prevent breaking changes if arguments are added or removed in future versions. For example, all arguments below after `*` must be passed by keyword:

```python
def __init__(self, *, eps=0.5, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, handle=None, verbose=False, output_type=None):
```

Finally, do not alter any input arguments to allow cloning of the estimator. See Sklearn's [section](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes) on instantiation for more info.

For example, the following `__init__` shows what **NOT** to do:
```python
def __init__(self, my_option="option1"):
   if (my_option == "option1"):
      self.my_option = 1
   else:
      self.my_option = 2
```

This will break cloning since the value of `self.my_option` is not a valid input to `__init__`.

### Estimator Array-Like Attributes

Any array attribute stored in an estimator needs to be convertable to the user's desired output type. To make it easier to store arrays in a class that derives from `Base`, the `cuml.common.array_descriptor.CumlArrayDescriptor` was created.

The `CumlArrayDescriptor` behaves different when accessed internally (from within on of `cuml`'s functions) vs. externally (as a user). Internally, it behaves exactly like a normal attribute, and will return the previous value set. Externally, the array will get converted to the users desired output type.

#### CumlArrayDescriptor Internal Functionality

To use the `CumlArrayDescriptor` in an estimator, the any array-like attributes need to be speficied by creating a `CumlArrayDescriptor` as a class variable. Other than that, developers can treat the attribute as they normally would.

Consider the following example estimator:
```python
import cupy as cp
import cuml
from cuml.common.array_descriptor import CumlArrayDescriptor

class TestEstimator(cuml.Base):

   # Class variables outside of any function
   my_cuml_array_ = CumlArrayDescriptor()
   my_cupy_array_ = CumlArrayDescriptor()
   my_other_array_ = CumlArrayDescriptor()

   def __init__(self, ...):

      # Initialize to None
      self.my_cuml_array_ = None

      # Init with a cupy array
      self.my_cupy_array_ = cp.zeros((10, 10))

   def fit(self, X):
      self._set_output_type(X)

      # Set my_cuml_array_ with a CumlArray
      self.my_cuml_array_, _, _, _ = input_to_cuml_array(X, order="K")

      # Access `my_cuml_array_` normally and set to another attribute
      # The internal type of my_other_array_ will be the result of
      # CumlArray + CuPy Array = Cupy Array
      self.my_other_array_ = self.my_cuml_array_ + self.my_cupy_array_

      return self
```

#### CumlArrayDescriptor External Functionality

Externally, when users read from any of `TestEstimator`'s attributes, they will be converted to the correct output type _lazily_ when the attribute is read from. For example:

```python
my_est = TestEstimator()

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
print(type(my_est.my_cuml_array_)) # Output: cudf

# Setting the global_output_type, overrides the estimator output_type attribute
with cuml.using_output_type("cupy"):
  print(type(my_est.my_cuml_array_)) # Output: cupy

# Once the global_output_type is restored, we return to the estimator output_type
print(type(my_est.my_cuml_array_)) # Output: Cudf. Using a cached value!
```

The internal representation of `CumlArrayDescriptor` is a `CumlArrayDescriptorMeta` object. To inspect the internal representation, the attribute value must be directly accessed from the estimator's `__dict__` (`getattr` and `__getattr__` will perform the conversion). For example:

```python
my_est = TestEstimator()
my_est.fit(cp.ones((10,)))

# Access the CumlArrayDescriptorMeta value directly. No array conversion will occur
print(my_est.__dict__["my_cuml_array_"])
# Output: CumlArrayDescriptorMeta(input_type='cupy', values={'cuml': <cuml.common.array.CumlArray object at 0x7fd39174ae20>, 'numpy': array([ 0,  1,  1,  2,  2, -1, -1, ...

# Values from CumlArrayDescriptorMeta can be specifically read
print(my_est.__dict__["my_cuml_array_"].input_type)
# Output: "cupy"

# The input value can be accessed
print(my_est.__dict__["my_cuml_array_"].get_input_value())
# Output: CumlArray ...
```

#### Internal Array-Like Conversion

Internally, developers can use the same `global_output_type` functionality to convert arrays as well. This has the same benefits of lazy conversion and caching as when descriptors are used externally. For example, we could add a `score()` function to `TestEstimator`:

```python
def score(self):

   # Set the global output type to numpy
   with cuml.using_output_type("numpy"):
      # Accessing my_cuml_array_ will return a numpy array and
      # the result can be returned directly
      return np.sum(self.my_cuml_array_, axis=0)
```

### Estimator Functions

In order for estimator functions to accept a wide variety of array types and correctly return the user's desired type, a fair amount of boiler plate code is involved. These common steps were often repetitive, inefficient, and incorrectly implemented.

To assit with this, a set of decorators have been created to wrap estimator functions (and all `cuml` API functions as well) and perform the boiler plate actions automatically.

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

While the above list of steps may seem excessive for every call, most functions follow this general form, but may skip a few steps depending on a couple of factors. For example, Step #3 is necessary on functions that modify the estimator's estimated attributes, such as `fit()`, but is not necessary for functions like `predict()` or `transform()`. And Step #5/6 are only necessary when returning array-like objects and are ommitted when returning any other type.

Functionally, you can think of these decorators equivalent to the following pseudocode:
```python
def my_func(self, X):
   with cuml.using_ouput_type("mirror"):
      with cupy.cuda.cupy_using_allocator(rmm.rmm_cupy_allocator):
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

Keep the above psuedocode in mind when working with these decorators since their goal is to replace many of these repetitive functions.

## Estimator Decorators

The two factors that determine which of the above steps will be performed are:
1. Function return type
2. Whether the function is on a class deriving from `Base`

The full set of descriptors can be organized by these two factors:

| Return Type-> | Array-Like | Sparse Array-Like | Generic | Any | 
| -----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| `Base`      | `@api_base_return_array` | `@api_base_return_sparse_array` |`@api_base_return_generic` | `@api_base_return_any` |
| Non-`Base`   | `@api_return_array` | `@api_return_sparse_array` | `@api_return_generic` | `@api_return_any` |

Simply choosing the decorator based off the return type and if the function is on `Base` will work most of the time. The decorator default options were designed to work on most estimator functions without much customization.

#### Decorator Functional Equivalent examples

The following section shows some simple comparisons between several functions before and after appling the decorators to illustrate what function calls they replace.

**Before: `predict()`**

```python
@with_cupy_rmm
def predict(self, X):
   # Determine the output_type
   out_type = self._get_output_type(X)

   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Convert the CudaArray to the desired output
   return X_m.to_output(output_type=out_type)
```

**After: `predict()`**

```python
@cuml.internals.api_base_return_array()
def predict(self, X):
   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Return CumlArray type. Will be converted automaticall
   return X_m
```

**Before: `fit()`**

```python
@with_cupy_rmm
def fit(self, X):
   # Set the base input attributes
   self._set_base_attributes(output_type=X, n_features=X)

   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

**After: `fit()`**

```python
@cuml.internals.api_base_return_any()
def fit(self, X):
   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

**Note:** These are very simple examples to illustrate which methods can be removed. Most of cuml Estimator functions are significantly more complex.

### Non-Standard Use Cases

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

## Automatic Decoration

Adding decorators to every estimator function just to use the decorator default values would be very repetitive and unnecessary. Because most of estimator functions follow a similar pattern, a new meta-class has been created to automatically decorate estimator functions based off their return type. This meta class will decorate functions according to a few rules:

1. If a functions has been manually decorated, it will not be automatically decoated
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

### Examples

**Before: `predict()`**
```python
@with_cupy_rmm
def predict(self, X, y):
   # Determine the output type and dtype
   out_type = self._get_output_type(y)

   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Convert the CudaArray to the desired output
   return X_m.to_output(output_type=out_type)
```

**After: `predict()`**
```python
def predict(self, X) -> CumlArray:
   # Convert to CumlArray
   X_m = input_to_cuml_array(X, order="K").array

   # Call a cuda function
   someCudaFunction(X_m.ptr)

   # Convert the CudaArray to the desired output
   return X_m
```

**Before: `fit()`**

```python
@with_cupy_rmm
def fit(self, X):
   # Set the base input attributes
   self._set_base_attributes(output_type=X, n_features=X)

   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

**After: `fit()`**

```python
def fit(self, X) -> "KMeans:
   self.coef_ = input_to_cuml_array(X, order="K").array

   return self
```

## Do's And Don'ts

### **Do:** Add Typing Information to Estimator Functions

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
def predict(self, X, convert_dtype=True) :
def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
def predict(self, start=0, end=None, level=None):
```

### **Do:** Return Array-Like Objects Directly

There is no need to convert the array type before returning it. Simply return any array-like object and it will be automatically converted

**Do this:**
```python
def predict(self, X, y) -> CumlArray:
   np_arr = np.ones((10,))

   return np_arr
```

**Not this:**
```python
def predict(self, X, y) -> CumlArray:
   np_arr = np.ones((10,))

   cuml_arr = input_to_cuml_array(X, order="K").array

   return cuml_arr.to_output(self._get_output_type(X))
```

### Example Decorator Use Cases

The following section shows some before and after examples of how the decorators should be used with default options.

**Example 1: 

Choosing the decorator by determining if the function is on `Base` and the return type will work in the majority of situations. The The following sections will cover the differences between each decorator, their default values, and when each should be used.

#### Decorator: `@api_base_return_any`, Return Type: Any, Derived From `Base`: Yes


<table>
   <thead>
      <tr>
         <th></th>
         <th></th>
         <th colspan=2>Defaults</th>
      </tr>
   </thead>
<tbody>
   <tr>
      <th rowspan=3>Can Set Input Values</th>
      <td rowspan=3><b>Yes</b></td>
      <td>output_type</td>
      <td>True</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <td>n_features</td>
      <td>True</td>
   </tr>
   <tr>
      <th rowspan=2>Can Get Output Values</th>
      <td rowspan=2><b>Yes</b></td>
      <td>output_type</td>
      <td>True</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <th>Converts Output</th>
      <td><b>Yes</b></td>
      <td></td>
      <td></td>
   </tr>
</tbody>
</table>

**When to Use:** Any function that returns something other than an array

**Common Functions:** `fit()`

#### Decorator: `@api_base_return_array`, Return Type: Array, Derived From `Base`: Yes

**When to Use:** Any function that returns an array-like object (`CumlArray`, `np.ndarray`, `cp.ndarray`, etc.)

<table>
   <thead>
      <tr>
         <th></th>
         <th></th>
         <th colspan=2>Defaults</th>
      </tr>
   </thead>
<tbody>
   <tr>
      <th rowspan=3>Can Set Input Values</th>
      <td rowspan=3><b>Yes</b></td>
      <td>output_type</td>
      <td>False</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <td>n_features</td>
      <td>False</td>
   </tr>
   <tr>
      <th rowspan=2>Can Get Output Values</th>
      <td rowspan=2><b>Yes</b></td>
      <td>output_type</td>
      <td>True</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <th>Converts Output</th>
      <td><b>Yes</b></td>
      <td></td>
      <td></td>
   </tr>
</tbody>
</table>

**Common Functions:** `predict()`, `transform()`, `inverse_transform()`

#### Decorator: `@api_base_return_generic`, Return Type: Generic, Derived From `Base`: Yes

**When to Use:** Any function that returns a Python "Generic" object that contains an array. Python "Generic" objects include `dict`, `list`, `tuple` and `typing.Union`. The generic type should contain an array-like object. For example, use this decorator when returning `typing.Tuple[CumlArray, float]`, but not `typing.Tuple[float, float]`. If the type does not contain an array-like object, use `@api_base_return_any`

<table>
   <thead>
      <tr>
         <th></th>
         <th></th>
         <th colspan=2>Defaults</th>
      </tr>
   </thead>
<tbody>
   <tr>
      <th rowspan=3>Can Set Input Values</th>
      <td rowspan=3><b>Yes</b></td>
      <td>output_type</td>
      <td>False</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <td>n_features</td>
      <td>False</td>
   </tr>
   <tr>
      <th rowspan=2>Can Get Output Values</th>
      <td rowspan=2><b>Yes</b></td>
      <td>output_type</td>
      <td>True</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <th>Converts Output</th>
      <td><b>Yes</b></td>
      <td colspan=2>Must be generic and contain array-like object</td>
   </tr>
</tbody>
</table>

**Common Functions:** `predict()`, `transform()`, `inverse_transform()`

#### Decorator: `@api_base_return_sparse_array`, Return Type: sparse array-like, Derived From `Base`: Yes

**When to Use:** Any function that returns a sparse array

<table>
   <thead>
      <tr>
         <th></th>
         <th></th>
         <th colspan=2>Defaults</th>
      </tr>
   </thead>
<tbody>
   <tr>
      <th rowspan=3>Can Set Input Values</th>
      <td rowspan=3><b>Yes</b></td>
      <td>output_type</td>
      <td>False</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <td>n_features</td>
      <td>False</td>
   </tr>
   <tr>
      <th rowspan=2>Can Get Output Values</th>
      <td rowspan=2><b>Yes</b></td>
      <td>output_type</td>
      <td>True</td>
   </tr>
   <tr>
      <td>output_dtype</td>
      <td>False</td>
   </tr>
   <tr>
      <th>Converts Output</th>
      <td><b>Yes</b></td>
      <td colspan=2>Must be generic and contain array-like object</td>
   </tr>
</tbody>
</table>

**Common Functions:** `predict()`, `transform()`, `inverse_transform()`