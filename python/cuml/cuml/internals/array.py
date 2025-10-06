#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import copy
import operator
from functools import cached_property
from typing import Tuple

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from numba import cuda
from numba.cuda import is_cuda_array as is_numba_array

import cuml.accel
import cuml.internals.nvtx as nvtx
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.logger import debug
from cuml.internals.mem_type import MemoryType, MemoryTypeError
from cuml.internals.memory_utils import class_with_cupy_rmm, with_cupy_rmm
from cuml.internals.output_utils import cudf_to_pandas

_specific_supported_types = (
    np.ndarray,
    cp.ndarray,
    cudf.Series,
    cudf.DataFrame,
    pd.Series,
    pd.DataFrame,
)


def _order_to_strides(order, shape, dtype):
    """
    Given memory order, shape and dtype, return expected strides
    """
    dtype = np.dtype(dtype)
    if order == "C":
        strides = (
            np.append(np.cumprod(np.array(shape[:0:-1]))[::-1], 1)
            * dtype.itemsize
        )
    elif order == "F":
        strides = np.cumprod(np.array([1, *shape[:-1]])) * dtype.itemsize
    else:
        raise ValueError(
            "Must specify strides or order, and order must"
            ' be one of "C" or "F"'
        )
    return strides


def _determine_memory_order(shape, strides, dtype, default="C"):
    """
    Given strides, shape and dtype for an array, return memory order

    If order is neither C nor F contiguous, return None. If array is both C and
    F contiguous, return default if given or 'C' otherwise.
    """
    if strides is None:
        return "C"
    if len(shape) < 2:
        return "C" if default in (None, "K") else default
    shape = np.array(shape)
    strides = np.array(strides)
    itemsize = np.dtype(dtype).itemsize
    c_contiguous = False
    f_contiguous = False
    if strides[-1] == itemsize:
        if np.all(strides[:-1] == shape[1:] * strides[1:]):
            c_contiguous = True
    if strides[0] == itemsize:
        if np.all(strides[1:] == shape[:-1] * strides[:-1]):
            f_contiguous = True
    if c_contiguous and f_contiguous:
        return "C" if default in (None, "K") else default
    elif c_contiguous:
        return "C"
    elif f_contiguous:
        return "F"
    return None


@class_with_cupy_rmm(ignore_pattern=["serialize"])
class CumlArray:

    """
    Array represents an abstracted array allocation. It can be instantiated by
    itself or can be instantiated by ``__cuda_array_interface__`` or
    ``__array_interface__`` compliant arrays, in which case it'll keep a
    reference to that data underneath. Also can be created from a pointer,
    specifying the characteristics of the array, in that case the owner of the
    data referred to by the pointer should be specified explicitly.

    Parameters
    ----------

    data : rmm.DeviceBuffer, cudf.Buffer, array_like, int, bytes, bytearray or\
           memoryview
        An array-like object or integer representing a
        device or host pointer to pre-allocated memory.
    owner : object, optional
        Python object to which the lifetime of the memory
        allocation is tied. If provided, a reference to this
        object is kept in this Buffer.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy or cupy data type.
    shape : int or tuple of ints, optional
        Shape of created array.
    order: string, optional
        Whether to create a F-major or C-major array.
    mem_type: {'host', 'device'}, optional
        Whether data are on host or device.
    validate: bool, default=None
        Whether or not to check final array attributes against input options.
        If None, validation will occur only for CumlArray input and input that
        does not implement the array interface protocol and for which
        additional options were explicitly specified.

    Attributes
    ----------

    ptr : int
        Pointer to the data
    size : int
        Size of the array data in bytes
    _owner : Python Object
        Object that owns the data of the array
    shape : tuple of ints
        Shape of the array
    order : {'F', 'C'}
        'F' or 'C' to indicate Fortran-major or C-major order of the array
    strides : tuple of ints
        Strides of the data
    mem_type : MemoryType
        Memory type for how data are stored
    __array_interface__ : dictionary
        ``__array_interface__`` to interop with other libraries. This
        attribute is only present if data are host-accessible.
    __cuda_array_interface__ : dictionary
        ``__cuda_array_interface__`` to interop with other libraries. This
        attribute is only present if data are device-accessible.

    Notes
    -----

    cuml Array is not meant as an end-user array library. It is meant for
    cuML/RAPIDS developer consumption. Therefore it contains the minimum
    functionality. Its functionality is hidden by base.pyx to provide
    automatic output format conversion so that the users see the important
    attributes in whatever format they prefer.

    Todo: support cuda streams in the constructor. See:
    https://github.com/rapidsai/cuml/issues/1712
    https://github.com/rapidsai/cuml/pull/1396

    """

    @nvtx.annotate(
        message="internals.CumlArray.__init__",
        category="utils",
        domain="cuml_python",
    )
    def __init__(
        self,
        data=None,
        index=None,
        owner=None,
        dtype=None,
        shape=None,
        order=None,
        strides=None,
        mem_type=None,
        validate=None,
    ):
        if dtype is not None:
            dtype = GlobalSettings().xpy.dtype(dtype)

        self._index = index
        if mem_type is not None:
            mem_type = MemoryType.from_str(mem_type)
        self._mem_type = mem_type

        if hasattr(data, "__cuda_array_interface__"):
            # using CuPy allows processing delayed array wrappers
            # like cumlarray without added complexity
            data = cp.asarray(data)
            # need to reshape if user requests specific shape
            if shape is not None:
                data = data.reshape(shape)
            self._array_interface = data.__cuda_array_interface__
            if mem_type in (None, MemoryType.mirror):
                self._mem_type = MemoryType.device
            self._owner = data
        else:  # Not a CUDA array object
            if hasattr(data, "__array_interface__"):
                self._array_interface = data.__array_interface__
                self._mem_type = MemoryType.host
                self._owner = data
            elif isinstance(data, (list, tuple)) and cuml.accel.enabled():
                # we accept lists and tuples in accel mode
                data = np.asarray(data)
                self._owner = data
                self._array_interface = data.__array_interface__
                self._mem_type = MemoryType.host
            else:  # Must construct array interface
                if dtype is None:
                    if hasattr(data, "dtype"):
                        dtype = data.dtype
                    else:
                        raise ValueError(
                            "Must specify dtype when data is passed as a"
                            " {}".format(type(data))
                        )
                if mem_type is None:
                    if GlobalSettings().memory_type in (
                        None,
                        MemoryType.mirror,
                    ):
                        raise ValueError(
                            "Must specify mem_type when data is passed as a"
                            " {}".format(type(data))
                        )
                    self._mem_type = GlobalSettings().memory_type

                if isinstance(data, int):
                    self._owner = owner
                else:

                    if self._mem_type is None:
                        cur_xpy = GlobalSettings().xpy
                    else:
                        cur_xpy = self._mem_type.xpy
                    # Assume integers are pointers. For everything else,
                    # convert it to an array and retry
                    try:
                        new_data = cur_xpy.frombuffer(data, dtype=dtype)
                    except TypeError:
                        new_data = cur_xpy.asarray(data, dtype=dtype)
                    if shape is not None:
                        new_order = order if order is not None else "C"
                        new_data = cur_xpy.reshape(
                            new_data, shape, order=new_order
                        )
                    if index is None:
                        try:
                            self._index = data.index
                        except AttributeError:
                            pass
                    return self.__init__(
                        data=new_data,
                        index=self._index,
                        owner=owner,
                        dtype=dtype,
                        shape=shape,
                        order=order,
                        mem_type=mem_type,
                    )

                if shape is None:
                    raise ValueError(
                        "shape must be specified when data is passed as a"
                        " pointer"
                    )
                if strides is None:
                    try:
                        if len(shape) == 0:
                            strides = None
                        elif len(shape) == 1:
                            strides = (dtype.itemsize,)
                    except TypeError:  # Shape given as integer
                        strides = (dtype.itemsize,)
                if strides is None:
                    strides = _order_to_strides(order, shape, dtype)

                self._array_interface = {
                    "shape": shape,
                    "strides": strides,
                    "typestr": dtype.str,
                    "data": (data, False),
                    "version": 3,
                }
        # Derive any information required for attributes that has not
        # already been derived
        if mem_type in (None, MemoryType.mirror):
            if self._mem_type in (None, MemoryType.mirror):
                raise ValueError(
                    "Could not infer memory type from input data. Pass"
                    " mem_type explicitly."
                )
            mem_type = self._mem_type

        if self._array_interface["strides"] is None:
            try:
                self._array_interface["strides"] = data.strides
            except AttributeError:
                self._array_interface["strides"] = strides

        if (
            isinstance(data, CumlArray)
            or not (
                hasattr(data, "__array_interface__")
                or hasattr(data, "__cuda_array_interface__")
            )
        ) and (dtype is not None and shape is not None and order is not None):
            self._array_interface["shape"] = shape
            self._array_interface["strides"] = strides
        else:
            if validate is None:
                validate = True

        array_strides = self._array_interface["strides"]
        if array_strides is not None:
            array_strides = np.array(array_strides)

        if (
            array_strides is None
            or len(array_strides) == 1
            or np.all(array_strides[1:] == array_strides[:-1])
        ) and order not in ("K", None):
            self._order = order
        else:
            self._order = _determine_memory_order(
                self._array_interface["shape"],
                self._array_interface["strides"],
                self._array_interface["typestr"],
                default=order,
            )

        # Validate final data against input arguments
        if validate:
            if mem_type != self._mem_type:
                raise MemoryTypeError(
                    "Requested mem_type inconsistent with input data object"
                )
            if (
                dtype is not None
                and dtype.str != self._array_interface["typestr"]
            ):
                raise ValueError(
                    "Requested dtype inconsistent with input data object"
                )
            if owner is not None and self._owner is not owner:
                raise ValueError(
                    "Specified owner object does not seem to match data"
                )
            if shape is not None:
                shape_arr = np.array(shape)
                if len(shape_arr.shape) == 0:
                    shape_arr = np.reshape(shape_arr, (1,))

                if not np.array_equal(
                    np.array(self._array_interface["shape"]), shape_arr
                ):
                    raise ValueError(
                        "Specified shape inconsistent with input data object"
                    )
            if (
                strides is not None
                and self._array_interface["strides"] is not None
                and not np.array_equal(
                    np.array(self._array_interface["strides"]),
                    np.array(strides),
                )
            ):
                raise ValueError(
                    "Specified strides inconsistent with input data object"
                )
            if order is not None and order != "K" and self._order != order:
                raise ValueError(
                    "Specified order inconsistent with array stride"
                )

    @property
    def ptr(self):
        return self._array_interface["data"][0]

    @cached_property
    def dtype(self):
        return self._mem_type.xpy.dtype(self._array_interface["typestr"])

    @property
    def mem_type(self):
        return self._mem_type

    @property
    def is_device_accessible(self):
        return self._mem_type.is_device_accessible

    @property
    def is_host_accessible(self):
        return self._mem_type.is_host_accessible

    @cached_property
    def size(self):
        return (
            np.prod(self._array_interface["shape"])
            * np.dtype(self._array_interface["typestr"]).itemsize
        )

    @property
    def order(self):
        return self._order

    @property
    def strides(self):
        return self._array_interface["strides"]

    @property
    def shape(self):
        return self._array_interface["shape"]

    @property
    def ndim(self):
        return len(self._array_interface["shape"])

    @cached_property
    def is_contiguous(self):
        return self.order in ("C", "F")

    # We use the index as a property to allow for validation/processing
    # in the future if needed
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def __cuda_array_interface__(self):
        if not self._mem_type.is_device_accessible:
            raise AttributeError(
                "Host-only array does not have __cuda_array_interface__"
            )
        return self._array_interface

    @property
    def __array_interface__(self):
        if not self._mem_type.is_host_accessible:
            raise AttributeError(
                "Device-only array does not have __array_interface__"
            )
        return self._array_interface

    @with_cupy_rmm
    def __getitem__(self, slice):
        return CumlArray(
            data=self._mem_type.xpy.asarray(self).__getitem__(slice)
        )

    @with_cupy_rmm
    def __iter__(self):
        arr = self._mem_type.xpy.asarray(self)
        yield from arr

    def __setitem__(self, slice, value):
        self._mem_type.xpy.asarray(self).__setitem__(slice, value)

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            return 0

    def _operator_overload(self, other, fn):
        return CumlArray(fn(self.to_output("array"), other))

    def __add__(self, other):
        return self._operator_overload(other, operator.add)

    def __sub__(self, other):
        return self._operator_overload(other, operator.sub)

    def __lt__(self, other):
        return self._operator_overload(other, operator.lt)

    def __le__(self, other):
        return self._operator_overload(other, operator.le)

    def __gt__(self, other):
        return self._operator_overload(other, operator.gt)

    def __ge__(self, other):
        return self._operator_overload(other, operator.ge)

    def __eq__(self, other):
        try:
            return self._operator_overload(other, operator.eq)
        except TypeError:
            return False

    def __or__(self, other):
        return self._operator_overload(other, operator.or_)

    def any(self):
        return self.to_output("array").any()

    def all(self):
        return self.to_output("array").all()

    def item(self):
        return self._mem_type.xpy.asarray(self).item()

    @nvtx.annotate(
        message="common.CumlArray.to_output",
        category="utils",
        domain="cuml_python",
    )
    def to_output(
        self, output_type="array", output_dtype=None, output_mem_type=None
    ):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:

            - 'array' - to cupy/numpy array depending on memory type
            - 'numba' - to numba device array
            - 'dataframe' - to cuDF/Pandas DataFrame depending on memory type
            - 'series' - to cuDF/Pandas Series depending on memory type
            - 'df_obj' - to cuDF/Pandas Series if array is single
              dimensional, to cuDF/Pandas Dataframe otherwise
            - 'cupy' - to cupy array
            - 'numpy' - to numpy array
            - 'cudf' - to cuDF Series/DataFrame depending on shape of data
            - 'pandas' - to Pandas Series/DataFrame depending on shape of data

        output_mem_type : {'host, 'device'}, optional
            Optionally convert array to given memory type. If `output_type`
            already indicates a specific memory type, `output_type` takes
            precedence. If the memory type is not otherwise indicated, the data
            are kept on their current device.

        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.

        """
        if output_type == "cupy":
            output_type = "array"
            output_mem_type = MemoryType.device
        elif output_type == "numpy":
            output_type = "array"
            output_mem_type = MemoryType.host
        elif output_type == "cudf":
            output_type = "df_obj"
            output_mem_type = MemoryType.device
        elif output_type == "pandas":
            output_type = "df_obj"
            output_mem_type = MemoryType.host

        if output_dtype is None:
            output_dtype = self.dtype

        if output_mem_type is None:
            output_mem_type = self._mem_type
        else:
            output_mem_type = MemoryType.from_str(output_mem_type)
            if output_mem_type == MemoryType.mirror:
                output_mem_type = self._mem_type

        if output_type == "df_obj":
            if len(self.shape) == 1:
                output_type = "series"
            elif len(self.shape) == 2 and self.shape[1] == 1:
                # It is convenient to coerce 2D arrays with second
                # dimension 1 to series, but we will not extend this to higher
                # dimensions
                output_type = "series"
            else:
                output_type = "dataframe"

        if output_type == "array":
            if output_mem_type == MemoryType.host:
                if self._mem_type == MemoryType.host:
                    return np.asarray(
                        self, dtype=output_dtype, order=self.order
                    )
                if isinstance(
                    self._owner, _specific_supported_types
                ) or "cuml" in str(type(self._owner)):
                    cp_arr = cp.asarray(
                        self, dtype=output_dtype, order=self.order
                    )
                else:
                    if self._owner is not None:
                        cp_arr = cp.asarray(
                            self._owner, dtype=output_dtype, order=self.order
                        )
                    else:
                        cp_arr = cp.asarray(
                            self, dtype=output_dtype, order=self.order
                        )
                return cp.asnumpy(
                    cp_arr,
                    order=self.order or "A",  # self.order may be None
                )
            return output_mem_type.xpy.asarray(
                self, dtype=output_dtype, order=self.order
            )

        elif output_type == "numba":
            return cuda.as_cuda_array(
                cp.asarray(self, dtype=output_dtype, order=self.order)
            )
        elif output_type == "series":
            if len(self.shape) == 2 and self.shape[1] == 1:
                arr = CumlArray(
                    self,
                    dtype=self.dtype,
                    order=self.order,
                    shape=(self.shape[0],),
                )
            else:
                arr = self

            if len(arr.shape) == 1:
                try:
                    if (
                        output_mem_type == MemoryType.host
                        and arr._mem_type != MemoryType.host
                    ):
                        return cudf_to_pandas(
                            cudf.Series(
                                arr, dtype=output_dtype, index=self.index
                            )
                        )
                    else:
                        return output_mem_type.xdf.Series(
                            arr, dtype=output_dtype, index=self.index
                        )
                except TypeError:
                    raise ValueError("Unsupported dtype for Series")
            else:
                raise ValueError(
                    "Only single dimensional arrays can be transformed to"
                    " Series."
                )
        elif output_type == "dataframe":
            arr = self.to_output(
                output_type="array",
                output_dtype=output_dtype,
                output_mem_type=output_mem_type,
            )
            if len(arr.shape) == 1:
                arr = arr.reshape(arr.shape[0], 1)
            if self.index is None:
                out_index = None
            elif (
                output_mem_type.is_device_accessible
                and not self.mem_type.is_device_accessible
            ):
                out_index = cudf.Index(self.index)
            elif (
                output_mem_type.is_host_accessible
                and not self.mem_type.is_host_accessible
            ):
                out_index = cudf_to_pandas(self.index)
            else:
                out_index = self.index
            if output_mem_type.is_device_accessible:
                # Do not convert NaNs to nulls in cuDF
                df_kwargs = {"nan_as_null": False}
            else:
                df_kwargs = {}
            try:
                return output_mem_type.xdf.DataFrame(
                    arr, index=out_index, **df_kwargs
                )
            except TypeError:
                raise ValueError("Unsupported dtype for DataFrame")

        return self

    @nvtx.annotate(
        message="common.CumlArray.host_serialize",
        category="utils",
        domain="cuml_python",
    )
    def host_serialize(self):
        mem_type = (
            self.mem_type
            if self.mem_type.is_host_accessible
            else MemoryType.host
        )
        return self.serialize(mem_type=mem_type)

    @classmethod
    def host_deserialize(cls, header, frames):
        assert all(not is_cuda for is_cuda in header["is-cuda"])
        obj = cls.deserialize(header, frames)
        return obj

    @nvtx.annotate(
        message="common.CumlArray.device_serialize",
        category="utils",
        domain="cuml_python",
    )
    def device_serialize(self):
        mem_type = (
            self.mem_type
            if self.mem_type.is_device_accessible
            else MemoryType.device
        )
        return self.serialize(mem_type=mem_type)

    @classmethod
    def device_deserialize(cls, header, frames):
        assert all(is_cuda for is_cuda in header["is-cuda"])
        obj = cls.deserialize(header, frames)
        return obj

    @nvtx.annotate(
        message="common.CumlArray.serialize",
        category="utils",
        domain="cuml_python",
    )
    def serialize(self, mem_type=None) -> Tuple[dict, list]:
        mem_type = self.mem_type if mem_type is None else mem_type
        header = {
            "constructor-kwargs": {
                "dtype": self.dtype.str,
                "shape": self.shape,
                "mem_type": mem_type.name,
            },
            "desc": self._array_interface,
            "frame_count": 1,
            "is-cuda": [mem_type.is_device_accessible],
            "lengths": [self.size],
        }
        frames = [self.to_output("array", output_mem_type=mem_type)]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize CumlArray with a single frame."
        ary = cls(data=frames[0], **header["constructor-kwargs"])

        if header["desc"]["shape"] != ary._array_interface["shape"]:
            raise ValueError(
                "Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {ary._array_interface['shape']}"
            )

        return ary.to_mem_type(GlobalSettings().memory_type)

    def __reduce_ex__(self, protocol):
        header, frames = self.host_serialize()
        return self.host_deserialize, (header, frames)

    @nvtx.annotate(
        message="common.CumlArray.to_host_array",
        category="utils",
        domain="cuml_python",
    )
    def to_mem_type(self, mem_type):
        return self.__class__(
            data=self.to_output("array", output_mem_type=mem_type),
            index=self.index,
            order=self.order,
            mem_type=MemoryType.from_str(mem_type),
            validate=False,
        )

    @nvtx.annotate(
        message="common.CumlArray.to_host_array",
        category="utils",
        domain="cuml_python",
    )
    def to_host_array(self):
        return self.to_output("numpy")

    @nvtx.annotate(
        message="common.CumlArray.to_host_array",
        category="utils",
        domain="cuml_python",
    )
    def to_device_array(self):
        return self.to_output("cupy")

    @classmethod
    @nvtx.annotate(
        message="common.CumlArray.empty",
        category="utils",
        domain="cuml_python",
    )
    def empty(cls, shape, dtype, order="F", index=None, mem_type=None):
        """
        Create an empty Array with an allocated but uninitialized DeviceBuffer

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        if mem_type is None:
            mem_type = GlobalSettings().memory_type

        return CumlArray(mem_type.xpy.empty(shape, dtype, order), index=index)

    @classmethod
    @nvtx.annotate(
        message="common.CumlArray.full", category="utils", domain="cuml_python"
    )
    def full(cls, shape, value, dtype, order="F", index=None, mem_type=None):
        """
        Create an Array with an allocated DeviceBuffer initialized to value.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """

        if mem_type is None:
            mem_type = GlobalSettings().memory_type
        return CumlArray(
            mem_type.xpy.full(shape, value, dtype, order), index=index
        )

    @classmethod
    @nvtx.annotate(
        message="common.CumlArray.zeros",
        category="utils",
        domain="cuml_python",
    )
    def zeros(
        cls, shape, dtype="float32", order="F", index=None, mem_type=None
    ):
        """
        Create an Array with an allocated DeviceBuffer initialized to zeros.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        return CumlArray.full(
            value=0,
            shape=shape,
            dtype=dtype,
            order=order,
            index=index,
            mem_type=mem_type,
        )

    @classmethod
    @nvtx.annotate(
        message="common.CumlArray.ones", category="utils", domain="cuml_python"
    )
    def ones(
        cls, shape, dtype="float32", order="F", index=None, mem_type=None
    ):
        """
        Create an Array with an allocated DeviceBuffer initialized to zeros.

        Parameters
        ----------
        dtype : data-type, optional
            Any object that can be interpreted as a numpy or cupy data type.
        shape : int or tuple of ints, optional
            Shape of created array.
        order: string, optional
            Whether to create a F-major or C-major array.
        """
        return CumlArray.full(
            value=1,
            shape=shape,
            dtype=dtype,
            order=order,
            index=index,
            mem_type=mem_type,
        )

    @classmethod
    @nvtx.annotate(
        message="common.CumlArray.from_input",
        category="utils",
        domain="cuml_python",
    )
    def from_input(
        cls,
        X,
        order="F",
        deepcopy=False,
        check_dtype=False,
        convert_to_dtype=False,
        check_mem_type=False,
        convert_to_mem_type=None,
        safe_dtype_conversion=True,
        check_cols=False,
        check_rows=False,
        fail_on_order=False,
        force_contiguous=True,
    ):
        """
        Convert input X to CumlArray.

        Acceptable input formats:

        * cuDF Dataframe - returns a deep copy always.
        * cuDF Series - returns by reference or a deep copy depending on
            `deepcopy`.
        * Numpy array - returns a copy in device always
        * cuda array interface compliant array (like Cupy) - returns a
            reference unless `deepcopy`=True.
        * numba device array - returns a reference unless deepcopy=True

        Parameters
        ----------

        X : cuDF.DataFrame, cuDF.Series, NumPy array, Pandas DataFrame, Pandas
            Series or any cuda_array_interface (CAI) compliant array like CuPy,
            Numba or pytorch.

        order: 'F', 'C' or 'K' (default: 'F')
            Whether to return a F-major ('F'),  C-major ('C') array or Keep
            ('K') the order of X. Used to check the order of the input. If
            fail_on_order=True, the method will raise ValueError, otherwise it
            will convert X to be of order `order` if needed.

        deepcopy: boolean (default: False)
            Set to True to always return a deep copy of X.

        check_dtype: np.dtype (default: False)
            Set to a np.dtype to throw an error if X is not of dtype
            `check_dtype`.

        convert_to_dtype: np.dtype (default: False)
            Set to a dtype if you want X to be converted to that dtype if it is
            not that dtype already.

        check_mem_type: {'host', 'device'} (default: False)
            Set to a value to throw an error if X is not of memory type
            `check_mem_type`.

        convert_to_mem_type: {'host', 'device'} (default: None)
            Set to a value if you want X to be converted to that memory type if
            it is not that memory type already. Set to False if you do not want
            any memory conversion. Set to None to use
            `cuml.global_settings.memory_type`.

        safe_convert_to_dtype: bool (default: True)
            Set to True to check whether a typecasting performed when
            convert_to_dtype is True will cause information loss. This has a
            performance implication that might be significant for very fast
            methods like FIL and linear models inference.

        check_cols: int (default: False)
            Set to an int `i` to check that input X has `i` columns. Set to
            False (default) to not check at all.

        check_rows: boolean (default: False)
            Set to an int `i` to check that input X has `i` rows. Set to
            False (default) to not check at all.

        fail_on_order: boolean (default: False)
            Set to True if you want the method to raise a ValueError if X is
            not of order `order`.

        force_contiguous: boolean (default: True)
            Set to True to force CumlArray produced to be contiguous. If `X` is
            non contiguous then a contiguous copy will be done.
            If False, and `X` doesn't need to be converted and is not
            contiguous, the underlying memory underneath the CumlArray will be
            non contiguous.  Only affects CAI inputs. Only affects CuPy and
            Numba device array views, all other input methods produce
            contiguous CumlArrays.

        Returns
        -------
        arr: CumlArray

            A new CumlArray

        """
        # Local to workaround circular imports
        from cuml.common.sparse_utils import is_sparse

        if is_sparse(X):
            # We don't support coercing sparse arrays to dense via this method.
            # Raising a NotImplementedError here lets us nicely error
            # for estimators that don't support sparse arrays without requiring
            # an additional external check. Otherwise they'd get an opaque error
            # for code below.
            raise NotImplementedError(
                "Sparse inputs are not currently supported for this method"
            )
        if convert_to_mem_type is None:
            convert_to_mem_type = GlobalSettings().memory_type
        else:
            convert_to_mem_type = (
                MemoryType.from_str(convert_to_mem_type)
                if convert_to_mem_type
                else convert_to_mem_type
            )
        if convert_to_dtype:
            convert_to_dtype = np.dtype(convert_to_dtype)
        # Provide fast-path for CumlArray input
        if (
            isinstance(X, CumlArray)
            and (
                not convert_to_mem_type
                or convert_to_mem_type == MemoryType.mirror
                or convert_to_mem_type == X.mem_type
            )
            and (not convert_to_dtype or convert_to_dtype == X.dtype)
            and (not force_contiguous or X.is_contiguous)
            and (order in ("K", None) or X.order == order)
            and not check_dtype
            and not check_mem_type
            and not check_cols
            and not check_rows
        ):
            if deepcopy:
                return copy.deepcopy(X)
            else:
                return X

        if hasattr(X, "__dask_graph__") and hasattr(X, "compute"):
            # TODO: Warn, but not when using dask_sql
            X = X.compute()

        index = getattr(X, "index", None)
        if index is not None:
            if convert_to_mem_type is MemoryType.host and isinstance(
                index, cudf.Index
            ):
                index = cudf_to_pandas(index)
            elif convert_to_mem_type is MemoryType.device and isinstance(
                index, pd.Index
            ):
                index = cudf.Index(index)

        if isinstance(X, cudf.Series):
            if X.null_count != 0:
                raise ValueError(
                    "Error: cuDF Series has missing/null values, "
                    "which are not supported by cuML."
                )

        if isinstance(X, cudf.DataFrame):
            X = X.to_cupy(copy=False)
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy(copy=False)
            # by default pandas converts to numpy 'C' major, which
            # does not keep the original order
            if order == "K":
                X = X.reshape(X.shape, order="F")
                order = "F"
        elif hasattr(X, "__dataframe__"):
            # temporarily use this codepath to avoid errors, substitute
            # usage of dataframe interchange protocol once ready.
            X = X.to_numpy()
            deepcopy = False
        elif isinstance(X, (list, tuple)) and cuml.accel.enabled():
            # we accept lists and tuples in accel mode
            X = np.asarray(X)
            deepcopy = False

        requested_order = (order, None)[fail_on_order]
        arr = cls(X, index=index, order=requested_order, validate=False)
        if deepcopy:
            arr = copy.deepcopy(arr)

        if convert_to_mem_type == MemoryType.mirror:
            convert_to_mem_type = arr.mem_type
        if convert_to_dtype:
            convert_to_dtype = arr.mem_type.xpy.dtype(convert_to_dtype)

        if check_dtype:
            # convert check_dtype to list if it's not a list already
            try:
                check_dtype = [
                    arr.mem_type.xpy.dtype(dtype) for dtype in check_dtype
                ]
            except TypeError:
                check_dtype = [arr.mem_type.xpy.dtype(check_dtype)]

            # if the input is in the desired dtypes, avoid conversion
            # otherwise, err if convert_dtype is false and input is not desired
            # dtype.
            if arr.dtype in check_dtype:
                convert_to_dtype = False
            else:
                if not convert_to_dtype:
                    raise TypeError(
                        f"Expected input to be of type in {check_dtype} but got"
                        f" {arr.dtype}"
                    )

        conversion_required = convert_to_dtype or (
            convert_to_mem_type and (convert_to_mem_type != arr.mem_type)
        )

        if conversion_required:
            convert_to_dtype = convert_to_dtype or None
            convert_to_mem_type = convert_to_mem_type or None
            if (
                safe_dtype_conversion
                and convert_to_dtype is not None
                and not arr.mem_type.xpy.can_cast(
                    arr.dtype, convert_to_dtype, casting="safe"
                )
            ):
                try:
                    target_dtype_range = arr.mem_type.xpy.iinfo(
                        convert_to_dtype
                    )
                except ValueError:
                    target_dtype_range = arr.mem_type.xpy.finfo(
                        convert_to_dtype
                    )
                if is_numba_array(X):
                    X = cp.asarray(X)
                if (
                    (X < target_dtype_range.min) | (X > target_dtype_range.max)
                ).any():
                    raise TypeError(
                        "Data type conversion on values outside"
                        " representable range of target dtype"
                    )
            arr = cls(
                arr.to_output(
                    output_dtype=convert_to_dtype,
                    output_mem_type=convert_to_mem_type,
                ),
                order=requested_order,
                index=index,
                validate=False,
            )

        make_copy = force_contiguous and not arr.is_contiguous

        if (
            not fail_on_order and order != arr.order and order != "K"
        ) or make_copy:
            if make_copy:
                data = arr.mem_type.xpy.array(
                    arr.to_output("array"), order=order
                )
            else:
                data = arr.mem_type.xpy.asarray(
                    arr.to_output("array"), order=order
                )

            arr = cls(data, index=index)

        n_rows = arr.shape[0]

        if len(arr.shape) > 1:
            n_cols = arr.shape[1]
        else:
            n_cols = 1

        if (n_cols == 1 or n_rows == 1) and len(arr.shape) == 2:
            order = "K"

        if order != "K" and arr.order != order:
            if order == "F":
                order_str = "column ('F')"
            elif order == "C":
                order_str = "row ('C')"
            else:
                order_str = f"UNKNOWN ('{order}')"
            if fail_on_order:
                raise ValueError(
                    f"Expected {order_str} major order but got something else."
                )
            else:
                debug(
                    f"Expected {order_str} major order but got something else."
                    " Converting data; this will result in additional memory"
                    " utilization."
                )

        if check_cols:
            if n_cols != check_cols:
                raise ValueError(
                    f"Expected {check_cols} columns but got {n_cols}"
                    " columns."
                )

        if check_rows:
            if n_rows != check_rows:
                raise ValueError(
                    f"Expected {check_rows} rows but got {n_rows}" " rows."
                )
        return arr


def array_to_memory_order(arr, default="C"):
    """
    Given an array-like object, determine its memory order

    If arr is C-contiguous, the string 'C' will be returned; if
    F-contiguous 'F'. If arr is neither C nor F contiguous, None will be
    returned. If an arr is both C and F contiguous, the indicated default
    will be returned. If a default of None or 'K' is given and the arr is both
    C and F contiguous, 'C' will be returned.
    """
    try:
        return arr.order
    except AttributeError:
        pass
    array_interface = getattr(
        arr,
        "__cuda_array_interface__",
        getattr(arr, "__array_interface__", False),
    )
    if not array_interface:
        return array_to_memory_order(CumlArray.from_input(arr, order="K"))

    strides = array_interface.get("strides", None)
    if strides is None:
        try:
            strides = arr.strides
        except AttributeError:
            pass
    return _determine_memory_order(
        array_interface["shape"],
        strides,
        array_interface["typestr"],
        default=default,
    )


def is_array_contiguous(arr):
    """Return true if array is C or F contiguous"""
    try:  # Fast path for CumlArray
        return arr.is_contiguous
    except AttributeError:
        pass
    try:  # Fast path for cupy/numpy arrays
        return arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]
    except (AttributeError, KeyError):
        return array_to_memory_order(arr) is not None


def elements_in_representable_range(arr, dtype):
    """Return true if all elements of the array can be represented in the
    available range of the given dtype"""
    arr = CumlArray.from_input(arr)
    dtype = arr.mem_type.xpy.dtype(dtype)
    try:
        dtype_range = arr.mem_type.xpy.iinfo(dtype)
    except ValueError:
        dtype_range = arr.mem_type.xpy.finfo(dtype)
    arr_xpy = arr.to_output("array")
    return not (
        ((arr_xpy < dtype_range.min) | (arr_xpy > dtype_range.max)).any()
    )
