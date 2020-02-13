#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
import rmm

from cuml.utils import rmm_cupy_ary
from numba import cuda
from numba.cuda.cudadrv.driver import driver


def row_matrix(df):
    """Compute the C (row major) version gpu matrix of df

    :param col_major: an `np.ndarray` or a `DeviceNDArrayBase` subclass.
        If already on the device, its stream will be used to perform the
        transpose (and to copy `row_major` to the device if necessary).

    """

    col_major = df.as_gpu_matrix(order='F')

    row_major = rmm_cupy_ary(cp.array, col_major, order='C')

    return cuda.as_cuda_array(row_major)


@cuda.jit
def gpu_zeros_1d(out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = 0


@cuda.jit
def gpu_zeros_2d(out):
    i, j = cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        out[i][j] = 0


def zeros(size, dtype, order='F'):
    """
    Return device array of zeros generated on device.
    """
    out = rmm.device_array(size, dtype=dtype, order=order)
    if isinstance(size, tuple):
        tpb = driver.get_device().MAX_THREADS_PER_BLOCK
        nrows = size[0]
        bpg = (nrows + tpb - 1) // tpb

        gpu_zeros_2d[bpg, tpb](out)

    elif size > 0:
        gpu_zeros_1d.forall(size)(out)

    return out


def device_array_from_ptr(ptr, shape, dtype, order='F', stride=None):
    """Create a DeviceNDArray from a device pointer

    Similar to numba.cuda.from_cuda_array_interface, difference is that we
    allow Fortran array order here.

    Parameters
    ----------
    ptr : int
        device pointer
    shape : tuple of ints
    dtype : type of the data
    order : 'C' or 'F'
    stride : tuple of ints
        Stride in bytes along each dimension the array. If it is left empty,
        then will be filled automatically using the order parameter
    """
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    if stride is None:
        stride = stride_from_order(shape, order, itemsize)
    size = cuda.driver.memory_size_from_info(shape, stride, itemsize)
    devptr = cuda.driver.get_devptr_for_active_ctx(ptr)
    data = cuda.driver.MemoryPointer(cuda.current_context(),
                                     devptr, size=size, owner=None)
    device_array = cuda.devicearray.DeviceNDArray(
        shape=shape, strides=stride, dtype=dtype, gpu_data=data)
    return device_array


def stride_from_order(shape, order, itemsize):
    if order != 'C' and order != 'F':
        raise ValueError('Order shall be either C or F')
    n = len(shape)
    stride = [None] * n
    stride[0] = itemsize
    for i in range(n - 1):
        stride[i+1] = stride[i] * shape[i]
    if order == 'C':
        stride.reverse()
    return tuple(stride)
