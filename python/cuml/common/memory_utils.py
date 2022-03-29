#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import contextlib
import functools
import operator
import re
from dataclasses import dataclass
from functools import wraps

import cuml
import cupy as cp
import numpy as np
import rmm
from cuml.common.import_utils import check_min_cupy_version
from numba import cuda as nbcuda

try:
    from cupy.cuda import using_allocator as cupy_using_allocator
except ImportError:
    try:
        from cupy.cuda.memory import using_allocator as cupy_using_allocator
    except ImportError:
        pass


@dataclass(frozen=True)
class ArrayInfo:
    """
    Calculate the necessary shape, order, stride and dtype of an array from an
    ``__array_interface__`` or ``__cuda_array_interface__``
    """
    shape: tuple
    order: str
    dtype: np.dtype
    strides: tuple

    @staticmethod
    def from_interface(interface: dict) -> "ArrayInfo":
        out_shape = interface['shape']
        out_type = np.dtype(interface['typestr'])
        out_order = "C"
        out_strides = None

        if interface.get('strides', None) is None:
            out_order = 'C'
            out_strides = _order_to_strides(out_order, out_shape, out_type)
        else:
            out_strides = interface['strides']
            out_order = _strides_to_order(out_strides, out_type)

        return ArrayInfo(shape=out_shape,
                         order=out_order,
                         dtype=out_type,
                         strides=out_strides)


def with_cupy_rmm(func):
    """

    Decorator to call CuPy functions with RMM memory management. Use it
    to decorate any function that will call CuPy functions. This will ensure
    that those calls use RMM for memory allocation instead of the default
    CuPy pool. Example:

    .. code-block:: python

        @with_cupy_rmm
        def fx(...):
            a = cp.arange(10) # uses RMM for allocation

    """

    if (func.__dict__.get("__cuml_rmm_wrapped", False)):
        return func

    @wraps(func)
    def cupy_rmm_wrapper(*args, **kwargs):
        with cupy_using_allocator(rmm.rmm_cupy_allocator):
            return func(*args, **kwargs)

    # Mark the function as already wrapped
    cupy_rmm_wrapper.__dict__["__cuml_rmm_wrapped"] = True

    return cupy_rmm_wrapper


def class_with_cupy_rmm(skip_init=False,
                        skip_private=True,
                        skip_dunder=True,
                        ignore_pattern: list = []):

    regex_list = ignore_pattern

    if (skip_private):
        # Match private but not dunder
        regex_list.append(r"^_(?!(_))\w+$")

    if (skip_dunder):
        if (not skip_init):
            # Make sure to not match __init__
            regex_list.append(r"^__(?!(init))\w+__$")
        else:
            # Match all dunder
            regex_list.append(r"^__\w+__$")
    elif (skip_init):
        regex_list.append(r"^__init__$")

    final_regex = '(?:%s)' % '|'.join(regex_list)

    def inner(klass):

        for attributeName, attribute in klass.__dict__.items():

            # Skip patters that dont match
            if (re.match(final_regex, attributeName)):
                continue

            if callable(attribute):

                # Passed the ignore patters. Wrap the function (will do nothing
                # if already wrapped)
                setattr(klass, attributeName, with_cupy_rmm(attribute))

            # Class/Static methods work differently since they are descriptors
            # (and not callable). Instead unwrap the function, and rewrap it
            elif (isinstance(attribute, classmethod)):
                unwrapped = attribute.__func__

                setattr(klass,
                        attributeName,
                        classmethod(with_cupy_rmm(unwrapped)))

            elif (isinstance(attribute, staticmethod)):
                unwrapped = attribute.__func__

                setattr(klass,
                        attributeName,
                        staticmethod(with_cupy_rmm(unwrapped)))

        return klass

    return inner


def rmm_cupy_ary(cupy_fn, *args, **kwargs):
    """

    Function to call CuPy functions with RMM memory management

    Parameters
    ----------
    cupy_fn : cupy function,
        CuPy function to execute, for example cp.array

    *args :
        Non keyword arguments to pass to the CuPy function

    **kwargs :
        Keyword named arguments to pass to the CuPy function


    .. note:: this function should be used if the result of cupy_fn creates
    a new array. Functions to create a new CuPy array by reference to
    existing device array (through __cuda_array_interface__) can be used
    directly.

    Examples
    --------

    >>> from cuml.common import rmm_cupy_ary
    >>> import cupy as cp
    >>>
    >>> # Get a new array filled with 0, column major
    >>> a = rmm_cupy_ary(cp.zeros, 5, order='F')
    >>> a
    array([0., 0., 0., 0., 0.])

    """

    # using_allocator was introduced in CuPy 7. Once 7+ is required,
    # this check can be removed alongside the else code path.
    if check_min_cupy_version("7.0"):
        with cupy_using_allocator(rmm.rmm_cupy_allocator):
            result = cupy_fn(*args, **kwargs)

    else:
        temp_res = cupy_fn(*args, **kwargs)
        result = \
            _rmm_cupy6_array_like(temp_res,
                                  order=_strides_to_order(temp_res.strides,
                                                          temp_res.dtype))
        cp.copyto(result, temp_res)

    return result


def _rmm_cupy6_array_like(ary, order):
    nbytes = np.ndarray(ary.shape,
                        dtype=ary.dtype,
                        strides=ary.strides,
                        order=order).nbytes
    memptr = rmm.rmm_cupy_allocator(nbytes)
    arr = cp.ndarray(ary.shape,
                     dtype=ary.dtype,
                     memptr=memptr,
                     strides=ary.strides,
                     order=order)
    return arr


def _strides_to_order(strides, dtype):
    # cuda array interface specification
    if strides is None:
        return 'C'
    if strides[0] == dtype.itemsize or len(strides) == 1:
        return 'F'
    elif strides[1] == dtype.itemsize:
        return 'C'
    else:
        raise ValueError("Invalid strides value for dtype")


def _order_to_strides(order, shape, dtype):
    itemsize = cp.dtype(dtype).itemsize
    if isinstance(shape, int):
        return (itemsize, )

    elif len(shape) == 0:
        return None

    elif len(shape) == 1:
        return (itemsize, )

    elif order == 'C':
        dim_minor = shape[1] * itemsize
        return (dim_minor, itemsize)

    elif order == 'F':
        dim_minor = shape[0] * itemsize
        return (itemsize, dim_minor)

    else:
        raise ValueError('Order must be "F" or "C". ')


def _get_size_from_shape(shape, dtype):
    """
    Calculates size based on shape and dtype, returns (None, None) if either
    shape or dtype are None
    """

    if shape is None or dtype is None:
        return (None, None)

    itemsize = cp.dtype(dtype).itemsize
    if isinstance(shape, int):
        size = itemsize * shape
        shape = (shape, )
    elif isinstance(shape, tuple):
        size = functools.reduce(operator.mul, shape)
        size = size * itemsize
    else:
        raise ValueError("Shape must be int or tuple of ints.")
    return (size, shape)


def _check_array_contiguity(ary):
    """
    Check if array-like ary is contioguous.

    Parameters
    ----------
    ary: __cuda_array_interface__ or __array_interface__ compliant array.
    """

    if hasattr(ary, 'ndim'):
        if ary.ndim == 1:
            return True

    # Use contiguity flags if present
    if hasattr(ary, 'flags'):
        if ary.flags['C_CONTIGUOUS'] or ary.flags['F_CONTIGUOUS']:
            return True
        else:
            return False

    # Check contiguity from shape and strides if not
    else:
        if hasattr(ary, "__array_interface__"):
            ary_interface = ary.__array_interface__

        elif hasattr(ary, "__cuda_array_interface__"):
            ary_interface = ary.__cuda_array_interface__

        else:
            raise TypeError("No array_interface attribute detected in input. ")

        # if the strides are not set or none, then the array is C-contiguous
        if 'strides' not in ary_interface or ary_interface['strides'] is None:
            return True

        shape = ary_interface['shape']
        strides = ary_interface['strides']
        dtype = cp.dtype(ary_interface['typestr'])
        order = _strides_to_order(strides, dtype)
        itemsize = cp.dtype(dtype).itemsize

        # We check if the strides jump on the non contiguous dimension
        # does not correspond to the array dimension size, which indicates
        # this is a view to a non contiguous array.
        if order == 'F':
            if (shape[0] * itemsize) != strides[1]:
                return False

        elif order == 'C':
            if (shape[1] * itemsize) != strides[0]:
                return False

        return True


def set_global_output_type(output_type):
    """
    Method to set cuML's single GPU estimators global output type.
    It will be used by all estimators unless overriden in their initialization
    with their own output_type parameter. Can also be overriden by the context
    manager method :func:`using_output_type`.

    Parameters
    ----------
    output_type : {'input', 'cudf', 'cupy', 'numpy'} (default = 'input')
        Desired output type of results and attributes of the estimators.

        * ``'input'`` will mean that the parameters and methods will mirror the
          format of the data sent to the estimators/methods as much as
          possible. Specifically:

          +---------------------------------------+--------------------------+
          | Input type                            | Output type              |
          +=======================================+==========================+
          | cuDF DataFrame or Series              | cuDF DataFrame or Series |
          +---------------------------------------+--------------------------+
          | NumPy arrays                          | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Pandas DataFrame or Series            | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Numba device arrays                   | Numba device arrays      |
          +---------------------------------------+--------------------------+
          | CuPy arrays                           | CuPy arrays              |
          +---------------------------------------+--------------------------+
          | Other `__cuda_array_interface__` objs | CuPy arrays              |
          +---------------------------------------+--------------------------+

        * ``'cudf'`` will return cuDF Series for single dimensional results and
          DataFrames for the rest.

        * ``'cupy'`` will return CuPy arrays.

        * ``'numpy'`` will return NumPy arrays.

    Examples
    --------

    >>> import cuml
    >>> import cupy as cp
    >>>
    >>> ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    >>> ary = cp.asarray(ary)
    >>> prev_output_type = cuml.global_settings.output_type
    >>> cuml.set_global_output_type('cudf')
    >>> dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    >>> dbscan_float.fit(ary)
    DBSCAN()
    >>>
    >>> # cuML output type
    >>> dbscan_float.labels_
    0    0
    1    1
    2    2
    dtype: int32
    >>> type(dbscan_float.labels_)
    <class 'cudf.core.series.Series'>
    >>> cuml.set_global_output_type(prev_output_type)

    Notes
    -----
    ``'cupy'`` and ``'numba'`` options (as well as ``'input'`` when using Numba
    and CuPy ndarrays for input) have the least overhead. cuDF add memory
    consumption and processing time needed to build the Series and DataFrames.
    ``'numpy'`` has the biggest overhead due to the need to transfer data to
    CPU memory.

    """
    if (isinstance(output_type, str)):
        output_type = output_type.lower()

    # Check for allowed types. Allow 'cuml' to support internal estimators
    if output_type not in [
            'numpy', 'cupy', 'cudf', 'numba', 'cuml', "input", None
    ]:
        # Omit 'cuml' from the error message. Should only be used internally
        raise ValueError('Parameter output_type must be one of "numpy", '
                         '"cupy", cudf", "numba", "input" or None')

    cuml.global_settings.output_type = output_type


@contextlib.contextmanager
def using_output_type(output_type):
    """
    Context manager method to set cuML's global output type inside a `with`
    statement. It gets reset to the prior value it had once the `with` code
    block is executer.

    Parameters
    ----------
    output_type : {'input', 'cudf', 'cupy', 'numpy'} (default = 'input')
        Desired output type of results and attributes of the estimators.

        * ``'input'`` will mean that the parameters and methods will mirror the
          format of the data sent to the estimators/methods as much as
          possible. Specifically:

          +---------------------------------------+--------------------------+
          | Input type                            | Output type              |
          +=======================================+==========================+
          | cuDF DataFrame or Series              | cuDF DataFrame or Series |
          +---------------------------------------+--------------------------+
          | NumPy arrays                          | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Pandas DataFrame or Series            | NumPy arrays             |
          +---------------------------------------+--------------------------+
          | Numba device arrays                   | Numba device arrays      |
          +---------------------------------------+--------------------------+
          | CuPy arrays                           | CuPy arrays              |
          +---------------------------------------+--------------------------+
          | Other `__cuda_array_interface__` objs | CuPy arrays              |
          +---------------------------------------+--------------------------+

        * ``'cudf'`` will return cuDF Series for single dimensional results and
          DataFrames for the rest.

        * ``'cupy'`` will return CuPy arrays.

        * ``'numpy'`` will return NumPy arrays.

    Examples
    --------

    >>> import cuml
    >>> import cupy as cp
    >>>
    >>> ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    >>> ary = cp.asarray(ary)
    >>>
    >>> with cuml.using_output_type('cudf'):
    ...     dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    ...     dbscan_float.fit(ary)
    ...
    ...     print("cuML output inside 'with' context")
    ...     print(dbscan_float.labels_)
    ...     print(type(dbscan_float.labels_))
    ...
    DBSCAN()
    cuML output inside 'with' context
    0    0
    1    1
    2    2
    dtype: int32
    <class 'cudf.core.series.Series'>
    >>> # use cuml again outside the context manager
    >>> dbscan_float2 = cuml.DBSCAN(eps=1.0, min_samples=1)
    >>> dbscan_float2.fit(ary)
    DBSCAN()
    >>>
    >>> # cuML default output
    >>> dbscan_float2.labels_
    array([0, 1, 2], dtype=int32)
    >>> type(dbscan_float2.labels_)
    <class 'cupy._core.core.ndarray'>

    """
    prev_output_type = cuml.global_settings.output_type
    try:
        set_global_output_type(output_type)
        yield prev_output_type
    finally:
        cuml.global_settings.output_type = prev_output_type


@with_cupy_rmm
def numba_row_matrix(df):
    """Compute the C (row major) version gpu matrix of df

    :param col_major: an `np.ndarray` or a `DeviceNDArrayBase` subclass.
        If already on the device, its stream will be used to perform the
        transpose (and to copy `row_major` to the device if necessary).

    """

    col_major = df.to_cupy()

    row_major = cp.array(col_major, order='C')

    return nbcuda.as_cuda_array(row_major)
