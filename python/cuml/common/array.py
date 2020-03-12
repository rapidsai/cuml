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

import cupy as cp
import numpy as np

from rmm import DeviceBuffer
from cudf.core import Buffer, Series, DataFrame
from cuml.utils import with_cupy_rmm
from cuml.utils.memory_utils import _get_size_from_shape
from cuml.utils.memory_utils import _order_to_strides
from cuml.utils.memory_utils import _strides_to_order
from numba import cuda


class CumlArray(Buffer):

    """
    Array represents an abstracted array allocation. It can be instantiated by
    itself, creating an rmm.DeviceBuffer underneath, or can be instantiated by
    __cuda_array_interface__ or __array_interface__ compliant arrays, in which
    case it'll keep a reference to that data underneath. Also can be created
    from a pointer, specifying the characteristics of the array, in that case
    the owner of the data referred to by the pointer should be specified
    explicitly.

    To standardize our code, please import this using:

    from cuml.common.array import Array as cumlArray

    Parameters
    ----------

    data : rmm.DeviceBuffer, cudf.Buffer, array_like, int, bytes, bytearrar or
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
        __cuda_array_interface__ to interop with other libraries.

    Object Methods
    --------------

    to_output : Convert the array to the appropriate output format.

    Class Methods
    -------------

    Array.empty : Create an empty array, allocating a DeviceBuffer.
    Array.full : Create an Array with allocated DeviceBuffer initialized with
        a particular value.
    Array.ones : Create an Array with allocated DeviceBuffer initialized with
        ones.
    Array.zeros : Create an Array with allocated DeviceBuffer initialized with
        zeros.

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

    def __init__(self, data=None, owner=None, dtype=None, shape=None,
                 order=None):

        # Checks of parameters
        if data is None:
            raise TypeError("To create an empty Array, use the class method" +
                            " Array.empty().")
        elif isinstance(data, memoryview):
            data = np.asarray(data)

        if _check_low_level_type(data):
            if dtype is None or shape is None or order is None:
                raise TypeError("Need to specify dtype, shape and order when" +
                                " creating an Array from " + type(data) + ".")
            detailed_construction = True
        elif dtype is not None and shape is not None and order is not None:
            detailed_construction = True
        else:
            detailed_construction = False

        ary_interface = False

        # Base class (Buffer) constructor call
        size, shape = _get_size_from_shape(shape, dtype)
        super(CumlArray, self).__init__(data=data, owner=owner, size=size)

        # Post processing of meta data
        if detailed_construction:
            self.shape = shape
            self.dtype = np.dtype(dtype)
            self.order = order
            self.strides = _order_to_strides(order, shape, dtype)

        elif hasattr(data, "__array_interface__"):
            ary_interface = data.__array_interface__

        elif hasattr(data, "__cuda_array_interface__"):
            ary_interface = data.__cuda_array_interface__

        else:
            raise TypeError("Unrecognized data type: %s" % str(type(data)))

        if ary_interface:
            self.shape = ary_interface['shape']
            self.dtype = np.dtype(ary_interface['typestr'])
            if ary_interface.get('strides', None) is None:
                self.order = 'C'
                self.strides = _order_to_strides(self.order, self.shape,
                                                 self.dtype)
            else:
                self.strides = ary_interface['strides']
                self.order = _strides_to_order(self.strides, self.dtype)

    def __getitem__(self, slice):
        return CumlArray(data=cp.asarray(self).__getitem__(slice))

    def __setitem__(self, slice, value):
        cp.asarray(self).__setitem__(slice, value)

    def __reduce__(self):
        return self.__class__, (self.to_output('numpy'),)

    def __len__(self):
        return self.shape[0]

    @property
    def __cuda_array_interface__(self):
        output = {
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.dtype.str,
            "data": (self.ptr, False),
            "version": 2,
        }
        return output

    @with_cupy_rmm
    def to_output(self, output_type='cupy'):
        """
        Convert array to output format

        Parameters
        ----------
        output_type : string
            Format to convert the array to. Acceptable formats are:
            'cupy' - to cupy array
            'numpy' - to numpy (host) array
            'numba' - to numba device array
            'dataframe' - to cuDF DataFrame
            'series' - to cuDF Series
            'cudf' - to cuDF Series if array is single dimensional, to
                DataFrame otherwise
        """

        # check to translate cudf to actual type converted
        if output_type == 'cudf':
            if len(self.shape) == 1:
                output_type = 'series'
            elif self.shape[1] == 1:
                output_type = 'series'
            else:
                output_type = 'dataframe'

        if output_type == 'cupy':
            return cp.asarray(self)

        elif output_type == 'numba':
            return cuda.as_cuda_array(self)

        elif output_type == 'numpy':
            return cp.asnumpy(cp.asarray(self), order=self.order)

        elif output_type == 'dataframe':
            if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                  np.uint64, np.float16]:
                mat = cuda.as_cuda_array(self)
                if len(mat.shape) == 1:
                    mat = mat.reshape(mat.shape[0], 1)
                return DataFrame.from_gpu_matrix(mat)
            else:
                raise ValueError('cuDF unsupported Array dtype')

        elif output_type == 'series':
            # check needed in case output_type was passed as 'series'
            # directly instead of as 'cudf'
            if len(self.shape) == 1:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=self.dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')
            elif self.shape[1] > 1:
                raise ValueError('Only single dimensional arrays can be \
                                 transformed to cuDF Series. ')
            else:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=self.dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')

    def serialize(self):
        header, frames = super(CumlArray, self).serialize()
        header["constructor-kwargs"] = {
            "dtype": self.dtype.str,
            "shape": self.shape,
            "order": self.order,
        }
        frames = [Buffer(f) for f in frames]
        return header, frames

    @classmethod
    def empty(cls, shape, dtype, order='F'):
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

        size, _ = _get_size_from_shape(shape, dtype)
        dbuf = DeviceBuffer(size=size)
        return CumlArray(data=dbuf, shape=shape, dtype=dtype, order=order)

    @classmethod
    def full(cls, shape, value, dtype, order='F'):
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
        size, _ = _get_size_from_shape(shape, dtype)
        dbuf = DeviceBuffer(size=size)
        cp.asarray(dbuf).view(dtype=dtype).fill(value)
        return CumlArray(data=dbuf, shape=shape, dtype=dtype,
                         order=order)

    @classmethod
    def zeros(cls, shape, dtype='float32', order='F'):
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
        return CumlArray.full(value=0, shape=shape, dtype=dtype, order=order)

    @classmethod
    def ones(cls, shape, dtype='float32', order='F'):
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
        return CumlArray.full(value=1, shape=shape, dtype=dtype, order=order)


def _check_low_level_type(data):
    if not (
        hasattr(data, "__array_interface__")
        or hasattr(data, "__cuda_array_interface__")
    ):
        return True
    elif isinstance(data, (DeviceBuffer, Buffer)):
        return True
    else:
        return False
