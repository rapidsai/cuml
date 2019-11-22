#
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
import math


map_kernel = cp.RawKernel(r'''
extern "C" __global__
void map_label(int *x, int x_n, int *labels, int n_labels) {
  
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= x_n) return;

  extern __shared__ int label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x) 
    label_cache[i] = labels[i];

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


validate_kernel = cp.RawKernel(r'''
extern "C" __global__
void validate_kernel(int *x, int x_n, int *labels, int n_labels, int *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= x_n) return;

  extern __shared__ int label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x) 
    label_cache[i] = labels[i];

  __syncthreads();

  int unmapped_label = x[tid];
  bool found = false;
  for(int i = 0; i < n_labels; i++) {
    if(label_cache[i] == unmapped_label) {
      found = true;
      break;
    }
  }

  if(!found) out[0] = 0;
}
''', 'validate_kernel')


inverse_map_kernel = cp.RawKernel(r'''
extern "C" __global__
void inverse_map_kernel(int *labels, int n_labels, int *x, int x_n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= x_n) return;

  extern __shared__ int label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x) 
    label_cache[i] = labels[i];

  __syncthreads();

  x[tid] = label_cache[x[tid]];
}
''', 'inverse_map_kernel')


def make_monotonic(labels, classes=None, copy=False):

    labels = cp.asarray(labels, dtype=cp.int32)

    if copy:
        labels = labels.copy()

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    if classes is None:
        classes = cp.unique(labels)

    smem = 4 * labels.shape[0]
    map_kernel((math.ceil(labels.shape[0] / 32),), (32, ),
               (labels,
                labels.shape[0],
                classes,
                classes.shape[0]),
               shared_mem=smem)

    return labels, classes


def check_labels(labels, classes):

    labels = cp.asarray(labels, dtype=cp.int32)
    classes = cp.asarray(classes, dtype=cp.int32)

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    valid = cp.array([1])

    # TODO: Support more dtypes
    smem = 4 * int(classes.shape[0])
    validate_kernel((math.ceil(labels.shape[0] / 32),), (32, ),
                    (labels, labels.shape[0], classes, classes.shape[0], valid),
                    shared_mem=smem)

    return valid[0] == 1


def invert_labels(labels, classes, copy=False):

    labels = cp.asarray(labels, dtype=cp.int32)
    classes = cp.asarray(classes, dtype=cp.int32)

    if copy:
        labels = labels.copy()

    smem = 4 * classes.shape[0]
    inverse_map_kernel((math.ceil(labels.shape[0] / 32),), (32,),
                       (classes, classes.shape[0],
                        labels, labels.shape[0]), shared_mem=smem)

    return labels

