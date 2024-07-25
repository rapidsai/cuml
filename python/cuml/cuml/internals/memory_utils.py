#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import functools
import operator
import re
from functools import wraps

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.device_support import GPU_ENABLED
from cuml.internals.mem_type import MemoryType
from cuml.internals.output_type import (
    INTERNAL_VALID_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES,
)
from cuml.internals.safe_imports import (
    cpu_only_import_from,
    gpu_only_import_from,
    UnavailableNullContext,
)

CudfSeries = gpu_only_import_from("cudf", "Series")
CudfDataFrame = gpu_only_import_from("cudf", "DataFrame")
cupy_using_allocator = gpu_only_import_from(
    "cupy.cuda", "using_allocator", alt=UnavailableNullContext
)
PandasSeries = cpu_only_import_from("pandas", "Series")
PandasDataFrame = cpu_only_import_from("pandas", "DataFrame")
rmm_cupy_allocator = gpu_only_import_from(
    "rmm.allocators.cupy", "rmm_cupy_allocator"
)


def set_global_memory_type(memory_type):
    GlobalSettings().memory_type = MemoryType.from_str(memory_type)


class using_memory_type:
    def __init__(self, mem_type):
        self.mem_type = mem_type
        self.prev_mem_type = None

    def __enter__(self):
        self.prev_mem_type = GlobalSettings().memory_type
        set_global_memory_type(self.mem_type)

    def __exit__(self, *_):
        set_global_memory_type(self.prev_mem_type)


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

    if func.__dict__.get("__cuml_rmm_wrapped", False):
        return func

    @wraps(func)
    def cupy_rmm_wrapper(*args, **kwargs):
        if GPU_ENABLED:
            with cupy_using_allocator(rmm_cupy_allocator):
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    # Mark the function as already wrapped
    cupy_rmm_wrapper.__dict__["__cuml_rmm_wrapped"] = True

    return cupy_rmm_wrapper


def class_with_cupy_rmm(
    skip_init=False,
    skip_private=True,
    skip_dunder=True,
    ignore_pattern: list = [],
):

    regex_list = ignore_pattern

    if skip_private:
        # Match private but not dunder
        regex_list.append(r"^_(?!(_))\w+$")

    if skip_dunder:
        if not skip_init:
            # Make sure to not match __init__
            regex_list.append(r"^__(?!(init))\w+__$")
        else:
            # Match all dunder
            regex_list.append(r"^__\w+__$")
    elif skip_init:
        regex_list.append(r"^__init__$")

    final_regex = "(?:%s)" % "|".join(regex_list)

    def inner(klass):

        for attributeName, attribute in klass.__dict__.items():

            # Skip patters that dont match
            if re.match(final_regex, attributeName):
                continue

            if callable(attribute):

                # Passed the ignore patters. Wrap the function (will do nothing
                # if already wrapped)
                setattr(klass, attributeName, with_cupy_rmm(attribute))

            # Class/Static methods work differently since they are descriptors
            # (and not callable). Instead unwrap the function, and rewrap it
            elif isinstance(attribute, classmethod):
                unwrapped = attribute.__func__

                setattr(
                    klass, attributeName, classmethod(with_cupy_rmm(unwrapped))
                )

            elif isinstance(attribute, staticmethod):
                unwrapped = attribute.__func__

                setattr(
                    klass,
                    attributeName,
                    staticmethod(with_cupy_rmm(unwrapped)),
                )

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

    if GPU_ENABLED:
        with cupy_using_allocator(rmm_cupy_allocator):
            result = cupy_fn(*args, **kwargs)
    else:
        result = cupy_fn(*args, **kwargs)

    return result


def _get_size_from_shape(shape, dtype):
    """
    Calculates size based on shape and dtype, returns (None, None) if either
    shape or dtype are None
    """

    if shape is None or dtype is None:
        return (None, None)

    itemsize = GlobalSettings().xpy.dtype(dtype).itemsize
    if isinstance(shape, int):
        size = itemsize * shape
        shape = (shape,)
    elif isinstance(shape, tuple):
        size = functools.reduce(operator.mul, shape)
        size = size * itemsize
    else:
        raise ValueError("Shape must be int or tuple of ints.")
    return (size, shape)


def set_global_output_type(output_type):
    """
    Method to set cuML's single GPU estimators global output type.
    It will be used by all estimators unless overridden in their initialization
    with their own output_type parameter. Can also be overridden by the context
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
    if isinstance(output_type, str):
        output_type = output_type.lower()

    # Check for allowed types. Allow 'cuml' to support internal estimators
    if (
        output_type is not None
        and output_type != "cuml"
        and output_type not in INTERNAL_VALID_OUTPUT_TYPES
    ):
        valid_output_types_str = ", ".join(
            [f"'{x}'" for x in VALID_OUTPUT_TYPES]
        )
        raise ValueError(
            f"output_type must be one of {valid_output_types_str}"
            f" or None. Got: {output_type}"
        )

    GlobalSettings().output_type = output_type


class using_output_type:
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
    >>> ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    >>> ary = cp.asarray(ary)
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
    >>> # cuML default output
    >>> dbscan_float2.labels_
    array([0, 1, 2], dtype=int32)
    >>> isinstance(dbscan_float2.labels_, cp.ndarray)
    True
    """

    def __init__(self, output_type):
        self.output_type = output_type

    def __enter__(self):
        self.prev_output_type = GlobalSettings().output_type
        set_global_output_type(self.output_type)
        return self.prev_output_type

    def __exit__(self, *_):
        GlobalSettings().output_type = self.prev_output_type


def determine_array_memtype(X):
    try:
        return X.mem_type
    except AttributeError:
        pass
    if hasattr(X, "__cuda_array_interface__"):
        return MemoryType.device
    if hasattr(X, "__array_interface__"):
        return MemoryType.host
    if isinstance(X, (CudfDataFrame, CudfSeries)):
        return MemoryType.device
    if isinstance(X, (PandasDataFrame, PandasSeries)):
        return MemoryType.host
    return None
