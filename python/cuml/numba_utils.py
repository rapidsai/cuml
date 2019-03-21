# Copyright (c) 2018, NVIDIA CORPORATION.
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

from numba import cuda
from numba.cuda.cudadrv.driver import driver
from librmm_cffi import librmm as rmm
import numpy as np


def row_matrix(df):
    """Compute the C (row major) version gpu matrix of df

    :param col_major: an `np.ndarray` or a `DeviceNDArrayBase` subclass.
        If already on the device, its stream will be used to perform the
        transpose (and to copy `row_major` to the device if necessary).

    To be replaced by CUDA ml-prim in upcoming version
    """

    cols = [df._cols[k] for k in df._cols]
    ncols = len(cols)
    nrows = len(df)
    dtype = cols[0].dtype

    col_major = df.as_gpu_matrix(order='F')
    row_major = rmm.device_array((nrows, ncols), dtype=dtype, order='C')

    threads_per_block = driver.get_device().MAX_THREADS_PER_BLOCK
    blocks_per_grid = (nrows + threads_per_block - 1) // threads_per_block
    col_offsets = rmm.to_device(np.zeros(threads_per_block, dtype=np.int32))

    @cuda.jit
    def kernel(_col_major, _col_offsets, _row_major):
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if tid >= nrows:
            return
        while _col_offsets[tid] < _col_major.shape[1]:
            col_idx = _col_offsets[tid]
            _row_major[tid, col_idx] = _col_major[tid, col_idx]
            _col_offsets[tid] += 1

    kernel[blocks_per_grid, threads_per_block](col_major, col_offsets, row_major)

    return row_major
