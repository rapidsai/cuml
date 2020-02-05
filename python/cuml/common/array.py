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
from cuml.utils.memory_utils import _strides_to_order, _get_size_from_shape, \
    _order_to_strides, rmm_cupy_ary
from numba import cuda


class Array(Buffer):

    def __init__(self, data=None, owner=None, dtype=None, shape=None,
                 order=None):
        if data is None:
            raise TypeError("To create an empty Array, use the class method \
                             Array.empty()")
        if isinstance(data, int):
            if dtype is None or shape is None or order is None:
                raise TypeError("Need to specify dtype, shape and order when \
                                creating an Array from a pointer.")

        if isinstance(data, DeviceBuffer) or isinstance(data, int):
            size, shape = _get_size_from_shape(shape, dtype)
            super(Array, self).__init__(data=data, owner=owner, size=size)
            self.shape = shape
            self.dtype = np.dtype(dtype)
            self.order = order
            self.strides = _order_to_strides(order, shape, dtype)

        elif hasattr(data, "__array_interface__") or \
                hasattr(data, "__cuda_array_interface__"):
            super(Array, self).__init__(data=data, owner=owner)
            self.shape = data.shape
            self.dtype = np.dtype(data.dtype)
            self.strides = data.strides
            self.order = _strides_to_order(data.strides, data.dtype)

        else:
            raise TypeError("Unrecognized data type.")

    def __getitem__(self, slice):
        return Array(data=cp.asarray(self).__getitem__(slice))

    def __setitem__(self, slice, value):
        cp.asarray(self).__setitem__(slice, value)

    def to_output(self, output_type='cupy'):
        if output_type == 'cupy':
            return cp.asarray(self)

        elif output_type == 'numba':
            return cuda.as_cuda_array(self)

        elif output_type == 'numpy':
            return np.array(cp.asnumpy(cp.asarray(self)), order=self.order)

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
            if len(self.shape) == 1:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=self.dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')
            elif self.shape[0] > 1 and self.shape[1] > 1:
                raise ValueError('Only single dimensional arrays can be \
                                 transformed to cuDF Series. ')
            else:
                if self.dtype not in [np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16]:
                    return Series(self, dtype=self.dtype)
                else:
                    raise ValueError('cuDF unsupported Array dtype')

    def __reduce__(self):
        return self.__class__, (self.to_output('numpy'),)

    @classmethod
    def empty(cls, shape, dtype, order='F'):
        size, _ = _get_size_from_shape(shape, dtype)
        dbuf = DeviceBuffer(size=size)
        return Array(data=dbuf, shape=shape, dtype=dtype, order=order)

    @classmethod
    def zeros(cls, shape, dtype='float32', order='F'):
        return Array(rmm_cupy_ary(cp.zeros, shape, order=order,
                                  dtype=dtype))

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
