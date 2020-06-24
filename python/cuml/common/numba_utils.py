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

# DEPRECATED: to be removed once full migration to CumlArray is done
# remaining usages: blobs.pyx, regression.pyx

from numba import cuda
from numba.cuda.cudadrv.driver import driver


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


def zeros(size, dtype, order="F"):
    """
    Return device array of zeros generated on device.
    """
    out = cuda.device_array(size, dtype=dtype, order=order)
    if isinstance(size, tuple):
        tpb = driver.get_device().MAX_THREADS_PER_BLOCK
        nrows = size[0]
        bpg = (nrows + tpb - 1) // tpb

        gpu_zeros_2d[bpg, tpb](out)

    elif size > 0:
        gpu_zeros_1d.forall(size)(out)

    return out
