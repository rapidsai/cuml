# Copyright (c) 2019, NVIDIA CORPORATION.
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

map_kernel = cp.RawKernel(r'''
extern "C" __global__
void map_label(int *x, int x_n, int *labels, int n_labels) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid > x_n) return;

  extern __shared__ int label_cache[];
  if(tid == 0) {
    for(int i = 0; i < n_labels; i++) label_cache[i] = labels[i];
  }

  __syncthreads();

  int unmapped_label = x[tid];
  for(int i = 0; i < n_labels; i++) {
    if(label_cache[i] == unmapped_label) {
      x[tid] = i;
      break;
    }
  }
}
''', 'map_label')


def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):

    n_classes = len(classes)

    classes = cp.array(classes)

    is_binary = True if n_classes == 1 and cp.unique(y) == 2 else False

    sorted_classes = cp.array(classes, dtype=cp.int32)

    col_ind = cp.array(y).copy().astype(cp.int32)
    row_ind = cp.arange(0, col_ind.shape[0], 1, dtype=cp.int32)
    val = cp.ones(row_ind.shape[0], dtype=cp.int32)

    smem = 4 * sorted_classes.shape[0]
    map_kernel((col_ind.shape[0] / 32,), (32, ),
               (col_ind, col_ind.shape[0], sorted_classes, sorted_classes.shape[0]),
               shared_mem=smem)

    print(str(col_ind))

    sp = cp.sparse.coo_matrix((val, (row_ind, col_ind)),
                              shape=(col_ind.shape[0],
                                     sorted_classes.shape[0]),
                              dtype=cp.float32)

    if sparse_output:
        return sp
    else:
        return sp.toarray().astype(cp.int32)
