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

import math
import cudf
import cupy as cp

import numba.cuda

map_kernel = cp.RawKernel(r'''
extern "C" __global__
void map_label(int *x, int x_n, int *labels, int n_labels) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= x_n) return;

  extern __shared__ int label_cache[];
  if(threadIdx.x == 0) {
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

validate_kernel = cp.RawKernel(r'''
extern "C" __global__
void validate_kernel(int *x, int x_n, int *labels, int n_labels, int *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid >= x_n) return;

  extern __shared__ int label_cache[];
  if(threadIdx.x == 0) {
    for(int i = 0; i < n_labels; i++) {
        label_cache[i] = labels[i];
    }
  }

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
  if(threadIdx.x == 0) {
    for(int i = 0; i < n_labels; i++) label_cache[i] = labels[i];
  }

  __syncthreads();

  x[tid] = label_cache[x[tid]];
}
''', 'inverse_map_kernel')


def _validate_labels(y, classes):

    valid = cp.array([1], dtype=cp.int32)

    smem = 4 * int(classes.shape[0])
    validate_kernel((math.ceil(y.shape[0] / 32),), (32, ),
                    (y, y.shape[0], classes, classes.shape[0], valid),
                    shared_mem=smem)

    return valid[0] == 1


def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):

    classes = cp.asarray(classes, dtype=cp.int32)

    col_ind = cp.asarray(y, dtype=cp.int32).copy()

    if not _validate_labels(col_ind, classes):
        raise ValueError("Unseen classes encountered in input")

    row_ind = cp.arange(0, col_ind.shape[0], 1, dtype=cp.int32)

    smem = 4 * classes.shape[0]
    map_kernel((math.ceil(col_ind.shape[0] / 32),), (32, ),
               (col_ind, col_ind.shape[0], classes, classes.shape[0]),
               shared_mem=smem)

    val = cp.full(row_ind.shape[0], pos_label, dtype=cp.int32)

    sp = cp.sparse.coo_matrix((val, (row_ind, col_ind)),
                              shape=(col_ind.shape[0],
                                     classes.shape[0]),
                              dtype=cp.float32)

    if sparse_output:
        sp = sp.tocsr()
        return sp
    else:

        arr = sp.toarray().astype(cp.int32)
        arr[arr == 0] = neg_label

        return arr


class LabelBinarizer(object):

    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        if neg_label >= pos_label:
            raise ValueError("neg_label=%s must be less "
                             "than pos_label=%s." % (neg_label, pos_label))

        if sparse_output and (pos_label == 0 or neg_label != 0):
            raise ValueError("Sparse binarization is only supported "
                             "with non-zero"
                             "pos_label and zero neg_label, got pos_label=%s "
                             "and neg_label=%s"
                             % (pos_label, neg_label))

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y):
        """Fit label binarizer`

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        self.classes_ = cp.unique(y).astype(cp.int32)

        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def transform(self, y):

        return label_binarize(y, self.classes_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label,
                              sparse_output=self.sparse_output)

    def inverse_transform(self, y, threshold=None):
        """
        Transform binary labels back to multi-class labels
        :param Y:
        :param threshold:
        :return:
        """
        y_mapped = cp.argmax(
            cp.asarray(y, dtype=cp.int32), axis=1).astype(cp.int32)

        smem = 4 * self.classes_.shape[0]

        inverse_map_kernel((math.ceil(y_mapped.shape[0] / 32),), (32,),
                           (self.classes_, self.classes_.shape[0],
                            y_mapped, y_mapped.shape[0]), shared_mem=smem)

        return y_mapped

    def __getstate__(self):
        state = self.__dict__.copy()
        state['classes_'] = cudf.Series(numba.cuda.to_device(self.classes_))
        return state

    def __setstate__(self, state):
        state['classes_'] = cp.asarray(state["classes_"].to_gpu_array(),
                                       dtype=cp.int32)
        self.__dict__.update(state)
