#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cuml
import cupy as cp
import functools
import numpy as np
import operator
import rmm

from cuml.common.import_utils import check_min_cupy_version
from functools import wraps
from numba import cuda as nbcuda

try:
    from cupy.cuda import using_allocator as cupy_using_allocator
except ImportError:
    try:
        from cupy.cuda.memory import using_allocator as cupy_using_allocator
    except ImportError:
        pass


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
    @wraps(func)
    def cupy_rmm_wrapper(*args, **kwargs):
        with cupy_using_allocator(rmm.rmm_cupy_allocator):
            return func(*args, **kwargs)

    return cupy_rmm_wrapper


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


    Note: this function should be used if the result of cupy_fn creates
    a new array. Functions to create a new CuPy array by reference to
    existing device array (through __cuda_array_interface__) can be used
    directly.

    Examples
    ---------

    .. code-block:: python

        from cuml.common import rmm_cupy_ary
        import cupy as cp

        # Get a new array filled with 0, column major
        a = rmm_cupy_ary(cp.zeros, 5, order='F')


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
        return (itemsize,)

    elif len(shape) == 1:
        return (itemsize,)

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
        shape = (shape,)
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
    manager method `using_output_type`

    Parameters
    ----------
    output_type : {'input', 'cudf', 'cupy', 'numpy'} (default = 'input')
        Desired output type of results and attributes of the estimators.

        'input' will mean that the parameters and methods will mirror the
        format of the data sent to the estimators/methods as much as
        possible. Specifically:

            Input type -> Output type

            cuDF DataFrame or Series -> cuDF DataFrame or Series

            NumPy arrays -> NumPy arrays

            Pandas DataFrame or Series -> NumPy arrays

            Numba device arrays -> Numba device arrays

            CuPy arrays -> CuPy arrays

            Other __cuda_array_interface__ objects -> CuPy arrays

        'cudf' will return cuDF Series for single dimensional results and
        DataFrames for the rest.

        'cupy' will return CuPy arrays.

        'numpy' will return NumPy arrays.

    Examples
    --------

    .. code-block:: python

        import cuml
        import cupy as cp

        ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
        ary = cp.asarray(ary)

        cuml.set_global_output_type('cudf'):
        dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
        dbscan_float.fit(ary)

        print("cuML output type")
        print(dbscan_float.labels_)
        print(type(dbscan_float.labels_))

    Output:

    .. code-block:: python

        cuML output type
        0    0
        1    1
        2    2
        dtype: int32
        <class 'cudf.core.series.Series'>

    Notes
    -----
    'cupy' and 'numba' options (as well as 'input' when using Numba and CuPy
    ndarrays for input) have the least overhead. cuDF add memory consumption
    and processing time needed to build the Series and DataFrames. 'numpy' has
    the biggest overhead due to the need to transfer data to CPU memory.

    """
    if isinstance(output_type, str):
        output_type = output_type.lower()
        if output_type in ['numpy', 'cupy', 'cudf', 'numba', 'input']:
            cuml.global_output_type = output_type
        else:
            raise ValueError('Parameter output_type must be one of ' +
                             '"series", "dataframe", cupy", "numpy", ' +
                             '"numba" or "input')
    else:
        raise ValueError('Parameter output_type must be one of "series" ' +
                         '"dataframe", cupy", "numpy", "numba" or "input')


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

        'input' will mean that the parameters and methods will mirror the
        format of the data sent to the estimators/methods as much as
        possible. Specifically:

            Input type -> Output type

            cuDF DataFrame or Series -> cuDF DataFrame or Series

            NumPy arrays -> NumPy arrays

            Pandas DataFrame or Series -> NumPy arrays

            Numba device arrays -> Numba device arrays

            CuPy arrays -> CuPy arrays

            Other __cuda_array_interface__ objects -> CuPy arrays

        'cudf' will return cuDF Series for single dimensional results and
        DataFrames for the rest.

        'cupy' will return CuPy arrays.

        'numpy' will return NumPy arrays.

    Examples
    --------

    .. code-block:: python

        import cuml
        import cupy as cp

        ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
        ary = cp.asarray(ary)

        with cuml.using_output_type('cudf'):
            dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
            dbscan_float.fit(ary)

            print("cuML output inside `with` context")
            print(dbscan_float.labels_)
            print(type(dbscan_float.labels_))

        # use cuml again outside the context manager
        dbscan_float2 = cuml.DBSCAN(eps=1.0, min_samples=1)
        dbscan_float2.fit(ary)

        print("cuML default output")
        print(dbscan_float2.labels_)
        print(type(dbscan_float2.labels_))

    Output:

    .. code-block:: python

        cuML output inside `with` context
        0    0
        1    1
        2    2
        dtype: int32
        <class 'cudf.core.series.Series'>


        cuML default output
        [0 1 2]
        <class 'cupy.core.core.ndarray'>

    """
    if isinstance(output_type, str):
        output_type = output_type.lower()
        if output_type in ['numpy', 'cupy', 'cudf', 'numba', 'input']:
            prev_output_type = cuml.global_output_type
            try:
                cuml.global_output_type = output_type
                yield
            finally:
                cuml.global_output_type = prev_output_type
        else:
            raise ValueError('Parameter output_type must be one of "series" ' +
                             '"dataframe", cupy", "numpy", "numba" or "input')
    else:
        raise ValueError('Parameter output_type must be one of "series" ' +
                         '"dataframe", cupy", "numpy", "numba" or "input')


@with_cupy_rmm
def numba_row_matrix(df):
    """Compute the C (row major) version gpu matrix of df

    :param col_major: an `np.ndarray` or a `DeviceNDArrayBase` subclass.
        If already on the device, its stream will be used to perform the
        transpose (and to copy `row_major` to the device if necessary).

    """

    col_major = df.as_gpu_matrix(order='F')

    row_major = cp.array(col_major, order='C')

    return nbcuda.as_cuda_array(row_major)
