# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

import numba
import math

from numba import cuda
from numba.cuda.cudadrv.driver import driver
from librmm_cffi import librmm as rmm
import numpy as np


def row_matrix(df):
    """Compute the C (row major) version gpu matrix of df

    :param col_major: an `np.ndarray` or a `DeviceNDArrayBase` subclass.
        If already on the device, its stream will be used to perform the
        transpose (and to copy `row_major` to the device if necessary).

    """

    cols = [df._cols[k] for k in df._cols]
    ncols = len(cols)
    nrows = len(df)
    dtype = cols[0].dtype

    col_major = df.as_gpu_matrix(order='F')

    row_major = gpu_major_converter(col_major, nrows, ncols, dtype,
                                    to_order='C')

    return row_major


def gpu_major_converter(original, nrows, ncols, dtype, to_order='C'):
    row_major = rmm.device_array((nrows, ncols), dtype=dtype, order=to_order)

    tpb = driver.get_device().MAX_THREADS_PER_BLOCK

    tile_width = int(math.pow(2, math.log(tpb, 2) / 2))
    tile_height = int(tpb / tile_width)

    tile_shape = (tile_height, tile_width + 1)

    # blocks and threads for the shared memory/tiled algorithm
    # see http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/ # noqa
    blocks = int((row_major.shape[1]) / tile_height + 1), \
        int((row_major.shape[0]) / tile_width + 1)

    threads = tile_height, tile_width

    # blocks per gpu for the general kernel
    bpg = (nrows + tpb - 1) // tpb

    if dtype == 'float32':
        dev_dtype = numba.float32

    else:
        dev_dtype = numba.float64

    @cuda.jit
    def general_kernel(input, output):
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if tid >= nrows:
            return
        _col_offset = 0
        while _col_offset < input.shape[1]:
            col_idx = _col_offset
            output[tid, col_idx] = input[tid, col_idx]
            _col_offset += 1

    @cuda.jit
    def shared_kernel(input, output):

        tile = cuda.shared.array(shape=tile_shape, dtype=dev_dtype)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x * cuda.blockDim.x
        by = cuda.blockIdx.y * cuda.blockDim.y
        y = by + tx
        x = bx + ty

        if by + ty < input.shape[0] and bx + tx < input.shape[1]:
            tile[ty, tx] = input[by + ty, bx + tx]
        cuda.syncthreads()
        if y < output.shape[0] and x < output.shape[1]:
            output[y, x] = tile[tx, ty]

    # check if we cannot call the shared memory kernel
    # block limits: 2**31-1 for x, 65535 for y dim of blocks
    if blocks[0] > 2147483647 or blocks[1] > 65535:
        general_kernel[bpg, tpb](original, row_major)

    else:
        shared_kernel[blocks, threads](original, row_major)

    return row_major


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
