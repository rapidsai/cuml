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
import math

from cuml.common import with_cupy_rmm
from cuml.common.kernel_utils import cuda_kernel_factory


def _map_l1_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l1 function."""

    map_kernel_str = r'''
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;


      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += fabs(data[i]);
      }


      if(sum == 0) return;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    '''
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l1_norm_kernel")


def _map_l2_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l2 function."""

    map_kernel_str = r'''
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += (data[i] * data[i]);
      }

      if(sum == 0) return;

      sum = sqrt(sum);

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    '''
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l2_norm_kernel")


@with_cupy_rmm
def csr_row_normalize_l1(X, inplace=True):
    """Row normalize for csr matrix using the l1 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l1_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel((math.ceil(X.shape[0] / 32),), (32,),
           (X.data, X.indices, X.indptr, X.shape[0]))

    return X


@with_cupy_rmm
def csr_row_normalize_l2(X, inplace=True):
    """Row normalize for csr matrix using the l2 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l2_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel((math.ceil(X.shape[0] / 32),), (32,),
           (X.data, X.indices, X.indptr, X.shape[0]))

    return X
