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
import math

from cuml.common.memory_utils import with_cupy_rmm
from cuml.common.kernel_utils import cuda_kernel_factory

cov_kernel_str = r'''
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, {0} *mean_y, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;
  
    if(rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] = \
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_y[cid];
}
'''

def _cov_kernel(dtype):
    return cuda_kernel_factory(cov_kernel_str,
                               (dtype,),
                               "map_labels_kernel")

@with_cupy_rmm
def cov(x, y, mean_x=None, mean_y=None, copy=False):
    if x.dtype != y.dtype:
        raise ValueError("X and Y must have same dtype (%s != %s)" %
                         (x.dtype, y.dtype))

    if x.shape != y.shape:
      raise ValueError("X and Y must have same shape %s != %s" %
                       (x.shape, y.shape))

    if (mean_x is not None and mean_y is not None):
      if mean_x.dtype != mean_y.dtype:
          raise ValueError("Mean of X and Mean of Y must have same dtype"
                           "(%s != %s)" % (mean_x.dtype, mean_y.dtype))

      if mean_x.shape != mean_y.shape:
          raise ValueError("Mean of X and Mean of Y must have same shape"
                           "%s != %s" % (mean_x.shape, mean_y.shape))

    gram_matrix = x.T.dot(y) * (1 / x.shape[0])

    if cp.sparse.issparse(gram_matrix):
      gram_matrix = gram_matrix.todense()

    if mean_x is None:
        mean_x = x.sum(axis=0) * (1 / x.shape[0])
    
    if mean_y is None:
        mean_y = y.sum(axis=0) * (1 / y.shape[0])
    
    if copy:
      cov_result = cp.zeros((gram_matrix.shape[0], gram_matrix.shape[1]),
                            dtype=gram_matrix.dtype)
    else:
      cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

    block_size = ((32, 32))
    grid_size = ((math.ceil(gram_matrix.shape[0] / 32),
                 math.ceil(gram_matrix.shape[1] / 32)))

    compute_cov(
        grid_size, block_size,
        (cov_result, gram_matrix, mean_x, mean_y, gram_matrix.shape[0])
    )

    return cov_result
