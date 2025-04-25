#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import math

import cupy as cp

from cuml.common.kernel_utils import cuda_kernel_factory


def _binarize_kernel(x_dtype):
    binarize_kernel_str = r"""({0} *x, float threshold, int x_n) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid >= x_n) return;

    {0} val = x[tid];
    if(val > threshold)
        val = 1;
    else
        val = 0;

    x[tid] = val;
    }"""
    return cuda_kernel_factory(
        binarize_kernel_str, (x_dtype,), "binarize_kernel"
    )


def binarize(x, threshold, copy=False):
    """
    Binarizes an array by converting values
    greater than a threshold to 1s and less
    than a threshold to 0s.

    Parameters
    ----------

    x : array-like
        Array to binarize
    threshold : float
        The cut-off point for values to be converted to 1s.
    copy : bool
        Should the operation be done in place or a copy made
    """

    arr = cp.asarray(x, dtype=x.dtype)

    if copy:
        arr = arr.copy()

    tpb = 512
    binarizer = _binarize_kernel(x.dtype)
    binarizer((math.ceil(arr.size / tpb),), (tpb,), (x, threshold, arr.size))

    return arr
