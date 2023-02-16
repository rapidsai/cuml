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

# TODO(wphicks): Handle serialization

import operator
import pickle
import warnings
try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType, MemoryTypeError
from cuml.internals.memory_utils import with_cupy_rmm
from cuml.internals.memory_utils import class_with_cupy_rmm
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import,
    gpu_only_import_from,
    null_decorator
)
from typing import Tuple

cudf = gpu_only_import('cudf')
cp = gpu_only_import('cupy')
np = cpu_only_import('numpy')
rmm = gpu_only_import('rmm')

cuda = gpu_only_import_from('numba', 'cuda')
CudfBuffer = gpu_only_import_from('cudf.core.buffer', 'Buffer')
DeviceBuffer = gpu_only_import_from('rmm', 'DeviceBuffer')
global_settings = GlobalSettings()
nvtx_annotate = gpu_only_import_from(
    'nvtx',
    'annotate',
    alt=null_decorator
)


@class_with_cupy_rmm(ignore_pattern=["serialize"])
class CumlArray():

    """
    Array represents an abstracted array allocation. It can be instantiated by
    itself, creating an rmm.DeviceBuffer underneath, or can be instantiated by
    ``__cuda_array_interface__`` or ``__array_interface__`` compliant arrays,
    in which case it'll keep a reference to that data underneath. Also can be
    created from a pointer, specifying the characteristics of the array, in
    that case the owner of the data referred to by the pointer should be
    specified explicitly.

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
    __cuda_array_interface__ : dictionary
        ``__cuda_array_interface__`` to interop with other libraries.

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

    @nvtx_annotate(message="internals.CumlArray.__init__", category="utils",
                   domain="cuml_python")
    def __init__(self,
                 data=None,
                 index=None,
                 owner=None,
                 dtype=None,
                 shape=None,
                 order=None,
                 strides=None,
                 mem_type=None):

        if mem_type is None:
            xpy = global_settings.xpy
        else:
            xpy = mem_type.xpy

        if dtype is not None:
            dtype = xpy.dtype(dtype)

        self._index = index
        self._mem_type = mem_type

        # Coerce data into an array interface and determine mem_type and owner
        # if necessary
        try:
            self._array_interface = data.__cuda_array_interface__
            if mem_type in (None, MemoryType.mirror):
                self._mem_type = MemoryType.device
            self._owner = data
        except AttributeError:  # Not a Cuda array object
            try:
                self._array_interface = data.__array_interface__
                self.mem_type = MemoryType.host
                self._owner = data
            except AttributeError:  # Must construct array interface

                if dtype is None:
                    raise ValueError(
                        'Must specify dtype when data is passed as a'
                        ' {}'.format(type(data))
                    )
                if isinstance(data, (CudfBuffer, DeviceBuffer)):
                    self._mem_type = MemoryType.device
                elif mem_type is None:
                    raise ValueError(
                        'Must specify mem_type when data is passed as a'
                        ' {}'.format(type(data))
                    )

                try:
                    data = data.ptr
                    if shape is None:
                        shape = (data.size,)
                    self._owner = data
                except AttributeError:  # Not a buffer object
                    pass
                if isinstance(data, int):
                    self._owner = owner
                else:
                    # Assume integers are pointers. For everything else,
                    # convert it to an array and retry
                    return self.__init__(
                        data=xpy.asarray(data, dtype=dtype),
                        index=index,
                        owner=owner,
                        dtype=dtype,
                        shape=shape,
                        order=order,
                        mem_type=mem_type
                    )

                if shape is None:
                    raise ValueError(
                        'shape must be specified when data is passed as a'
                        ' pointer'
                    )
                if strides is None:
                    try:
                        if len(shape) == 0:
                            strides = None
                        elif len(shape) == 1:
                            strides == (dtype.itemsize,)
                    except TypeError:  # Shape given as integer
                        strides = (dtype.itemsize,)
                if strides is None:
                    if order == 'C':
                        strides = xpy.cumprod(shape[:0:-1])[::-1].append(
                            1
                        ) * dtype.itemsize
                    elif order == 'F':
                        strides = (xpy.cumprod(
                            (1, *shape[:-1])
                        ) * dtype.itemsize)
                    else:
                        raise ValueError(
                            'Must specify strides or order, and order must'
                            ' be one of "C" or "F"'
                        )

                self._array_interface = {
                    'shape': shape,
                    'strides': strides,
                    'typestr': dtype.str,
                    'data': (data, False),
                    'version': 3
                }
        # Derive any information required for attributes that has not
        # already been derived
        if mem_type in (None, MemoryType.mirror):
            if self._mem_type in (None, MemoryType.mirror):
                raise ValueError(
                    'Could not infer memory type from input data. Pass'
                    ' mem_type explicitly.'
                )
            mem_type = self._mem_type

        if (
            self._array_interface['strides'] is None or
            len(self._array_interface['strides']) == 1 or
            xpy.all(
                self._array_interface['strides'][1:]
                >= self._array_interface['strides'][:-1]
            )
        ):
            self._order = 'C'
        elif xpy.all(
            self._array_interface['strides'][1:]
            <= self._array_interface['strides'][:-1]
        ):
            self._order = 'F'
        else:
            self._order = None

        # Validate final data against input arguments
        if mem_type != self._mem_type:
            raise MemoryTypeError(
                'Requested mem_type inconsistent with input data object'
            )
        if (
            dtype is not None and dtype.str !=
            self._array_interface['typestr']
        ):
            raise ValueError(
                'Requested dtype inconsistent with input data object'
            )
        if owner is not None and self._owner is not owner:
            raise ValueError(
                'Specified owner object does not seem to match data'
            )
        if shape is not None and not xpy.array_equal(
                self._array_interface['shape'], shape):
            raise ValueError(
                'Specified shape inconsistent with input data object'
            )
        if shape is not None and not xpy.array_equal(
                self._array_interface['shape'], shape):
            raise ValueError(
                'Specified shape inconsistent with input data object'
            )
        if strides is not None and not xpy.array_equal(
                self._array_interface['strides'], strides):
            raise ValueError(
                'Specified strides inconsistent with input data object'
            )
        if order is not None and self._order != order:
            raise ValueError(
                'Specified order inconsistent with array stride'
            )

    @property
    def ptr(self):
        return self._array_interface['data'][0]

    @property
    @cache
    def size(self):
        xpy = self._mem_type.xpy
        return xpy.product(
            self._array_interface['shape']
        ) * xpy.dtype(self._array_interface['typestr'])

    @property
    def order(self):
        return self._order

    @property
    def strides(self):
        return self._array_interface['strides']

    @property
    def shape(self):
        return self._array_interface['shape']

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
                'Host-only array does not have __cuda_array_interface__'
            )
        return self._array_interface

    @property
    def __array_interface__(self):
        if not self._mem_type.is_host_accessible:
            raise AttributeError(
                'Device-only array does not have __array_interface__'
            )
        return self._array_interface

    @with_cupy_rmm
    def __getitem__(self, slice):
        return CumlArray(
            data=self._mem_type.xpy.asarray(self).__getitem__(slice)
        )

    def __setitem__(self, slice, value):
        self._mem_type.xpy.asarray(self).__setitem__(slice, value)

    def __len__(self):
        return self.shape[0]

    def _operator_overload(self, other, fn):
        return CumlArray(fn(self.to_output('array'), other))

    def __add__(self, other):
        return self._operator_overload(other, operator.add)

    def __sub__(self, other):
        return self._operator_overload(other, operator.sub)

    def item(self):
        return self._mem_type.xpy.asarray(self).item()

    @nvtx_annotate(message="common.CumlArray.to_output", category="utils",
                   domain="cuml_python")
    def to_output(
            self,
            output_type='array',
            output_dtype=None,
            output_mem_type=None):
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

        output_dtype : string, optional
            Optionally cast the array to a specified dtype, creating
            a copy if necessary.

        """
        if output_type == 'cupy':
            warnings.warn(
                '"cupy" is a deprecated output type for CumlArray. Use'
                ' output_type "array" with output_mem_type "device" instead.',
                DeprecationWarning
            )
            output_type = 'array'
            output_mem_type = MemoryType.device
        elif output_type == 'numpy':
            warnings.warn(
                '"numpy" is a deprecated output type for CumlArray. Use'
                ' output_type "array" with output_mem_type "host" instead.',
                DeprecationWarning
            )
            output_type = 'array'
            output_mem_type = MemoryType.host
        elif output_type == 'cudf':
            warnings.warn(
                '"cudf" is a deprecated output type for CumlArray. Use'
                ' output_type "df_obj" with output_mem_type "device" instead.',
                DeprecationWarning
            )
            output_type = 'df_obj'
            output_mem_type = MemoryType.device
        if output_dtype is None:
            output_dtype = self.dtype

        if output_mem_type is None:
            output_mem_type = self._mem_type
        else:
            output_mem_type = MemoryType.from_str(output_mem_type)
            if output_mem_type == MemoryType.mirror:
                output_mem_type = self._mem_type

        if output_type == 'df_obj':
            if len(self.shape) == 1:
                output_type = 'series'
            elif self.shape[1] == 1:
                output_type = 'series'
            else:
                output_type = 'dataframe'

        if output_type == 'array':
            if output_mem_type == MemoryType.host:
                if self._mem_type == MemoryType.host:
                    return np.asarray(self, dtype=output_dtype)
                return cp.asnumpy(
                    cp.asarray(self, dtype=output_dtype, order=self.order)
                )
            return output_mem_type.xpy.asarray(
                self, dtype=output_dtype
            )

        elif output_type == 'numba':
            return cuda.as_cuda_array(cp.asarray(self, dtype=output_dtype))
        elif output_type == 'series':
            if (
                len(self.shape) == 1 or
                (len(self.shape) == 2 and self.shape[1] == 1)
            ):
                try:
                    if (
                        output_mem_type == MemoryType.host
                        and self._mem_type != MemoryType.host
                    ):
                        return cudf.Series(
                            self,
                            dtype=output_dtype,
                            index=self.index
                        )
                    else:
                        return output_mem_type.xdf.Series(
                            self,
                            dtype=output_dtype,
                            index=self.index
                        )
                except TypeError:
                    raise ValueError('Unsupported dtype for Series')
            else:
                raise ValueError(
                    'Only single dimensional arrays can be transformed to'
                    ' Series.'
                )
        elif output_type == 'dataframe':
            arr = self.to_output(
                output_type='array',
                output_dtype=output_dtype,
                output_mem_type=output_mem_type
            )
            if len(arr.shape) == 1:
                arr = arr.reshape(arr.shape[0], 1)
            try:
                return output_mem_type.xdf.DataFrame(
                    arr, index=self.index
                )
            except TypeError:
                raise ValueError('Unsupported dtype for Series')

        return self

    @nvtx_annotate(message="common.CumlArray.host_serialize", category="utils",
                   domain="cuml_python")
    def host_serialize(self):
        header, frames = self.device_serialize()
        header["writeable"] = len(frames) * (None,)
        frames = [
            f.to_host_array().data if c else memoryview(f)
            for c, f in zip(header["is-cuda"], frames)
        ]
        return header, frames

    @classmethod
    def host_deserialize(cls, header, frames):
        frames = [
            rmm.DeviceBuffer.to_device(f) if c else f
            for c, f in zip(header["is-cuda"], map(memoryview, frames))
        ]
        obj = cls.device_deserialize(header, frames)
        return obj

    def device_serialize(self):
        header, frames = self.serialize()
        assert all(
            isinstance(f, (CumlArray, memoryview))
            for f in frames
        )
        header["type-serialized"] = pickle.dumps(type(self))
        header["is-cuda"] = [
            hasattr(f, "__cuda_array_interface__") for f in frames
        ]
        header["lengths"] = [f.nbytes for f in frames]
        return header, frames

    @classmethod
    def device_deserialize(cls, header, frames):
        typ = pickle.loads(header["type-serialized"])
        frames = [
            CumlArray(f) if c else memoryview(f)
            for c, f in zip(header["is-cuda"], frames)
        ]
        assert all(
            (isinstance(f._owner, rmm.DeviceBuffer))
            if c
            else (isinstance(f, memoryview))
            for c, f in zip(header["is-cuda"], frames)
        )
        obj = typ.deserialize(header, frames)

        return obj

    @nvtx_annotate(message="common.CumlArray.serialize", category="utils",
                   domain="cuml_python")
    def serialize(self) -> Tuple[dict, list]:
        header, frames = super().serialize()
        header["constructor-kwargs"] = {
            "dtype": self.dtype.str,
            "shape": self.shape,
            "order": self.order,
        }
        frames = [CumlArray(f) for f in frames]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list):
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize CumlArray with a single frame."
        ary = cls(frames[0], **header["constructor-kwargs"])

        if header["desc"]["shape"] != ary.__cuda_array_interface__["shape"]:
            raise ValueError(
                f"Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {ary.__cuda_array_interface__['shape']}"
            )

        return ary

    def __reduce_ex__(self, protocol):
        header, frames = self.host_serialize()
        frames = [f.obj for f in frames]
        return self.host_deserialize, (header, frames)

    @nvtx_annotate(message="common.CumlArray.copy", category="utils",
                   domain="cuml_python")
    def copy(self) -> CudfBuffer:
        """
        Create a new Buffer containing a copy of the data contained
        in this Buffer.
        """
        from rmm._lib.device_buffer import copy_device_to_ptr
        # TODO(wphicks): Generalize this for host/device

        out = CudfBuffer.empty(size=self.size)
        copy_device_to_ptr(self.ptr, out.ptr, self.size)
        return out

    @nvtx_annotate(message="common.CumlArray.to_host_array", category="utils",
                   domain="cuml_python")
    def to_host_array(self):
        # TODO(wphicks): Streamline this and generalize
        data = np.empty((self.size,), "u1")
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, data)
        return data

    @classmethod
    @nvtx_annotate(message="common.CumlArray.empty", category="utils",
                   domain="cuml_python")
    def empty(cls,
              shape,
              dtype,
              order='F',
              index=None,
              mem_type=None):
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
            mem_type = global_settings.memory_type

        return CumlArray(
            mem_type.xpy.empty(shape, dtype, order), index=index
        )

    @classmethod
    @nvtx_annotate(message="common.CumlArray.full", category="utils",
                   domain="cuml_python")
    def full(cls,
             shape,
             value,
             dtype,
             order='F',
             index=None,
             mem_type=None):
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
            mem_type = global_settings.memory_type
        return CumlArray(
            mem_type.xpy.full(shape, value, dtype, order), index=index
        )

    @classmethod
    @nvtx_annotate(message="common.CumlArray.zeros", category="utils",
                   domain="cuml_python")
    def zeros(cls,
              shape,
              dtype='float32',
              order='F',
              index=None,
              mem_type=None):
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
        return CumlArray.full(value=0, shape=shape, dtype=dtype, order=order,
                              index=index, mem_type=mem_type)

    @classmethod
    @nvtx_annotate(message="common.CumlArray.ones", category="utils",
                   domain="cuml_python")
    def ones(cls,
             shape,
             dtype='float32',
             order='F',
             index=None,
             mem_type=None):
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
        return CumlArray.full(value=1, shape=shape, dtype=dtype, order=order,
                              index=index, mem_type=mem_type)
