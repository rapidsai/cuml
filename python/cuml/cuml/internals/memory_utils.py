#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import functools
import operator

import cudf
import pandas as pd

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType
from cuml.internals.output_type import (
    INTERNAL_VALID_OUTPUT_TYPES,
    VALID_OUTPUT_TYPES,
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
    if isinstance(X, (cudf.DataFrame, cudf.Series)):
        return MemoryType.device
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return MemoryType.host
    return None


def cuda_ptr(X):
    """Returns a pointer to a backing device array, or None if not a device array"""
    if (interface := getattr(X, "__cuda_array_interface__", None)) is not None:
        return interface["data"][0]
    return None
