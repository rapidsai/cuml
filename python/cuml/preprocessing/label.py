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

import cudf
import cupy as cp

import numba.cuda

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

validate_kernel = cp.RawKernel(r'''
extern "C" __global__
void validate_kernel(int *x, int x_n, int *labels, int n_labels, bool *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid > x_n) return;

  extern __shared__ int label_cache[];
  if(tid == 0) {
    for(int i = 0; i < n_labels; i++) label_cache[i] = labels[i];
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
  
  if(!found) out[0] = false;
}
''', 'validate_kernel')

inverse_map_kernel = cp.RawKernel(r'''
extern "C" __global__
void inverse_map_kernel(int *labels, int n_labels, int *x, int x_n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid > x_n) return;

  extern __shared__ int label_cache[];
  if(tid == 0) {
    for(int i = 0; i < n_labels; i++) label_cache[i] = labels[i];
  }

  __syncthreads();

  x[tid] = label_cache[x[tid]];
}
''', 'inverse_map_kernel')


def _validate_labels(y, classes):

    valid = cp.array([True], dtype=cp.bool_)

    smem = 4 * classes.shape[0]
    map_kernel((y.shape[0] / 32,), (32, ),
               (classes, classes.shape[0], y, y.shape[0]),
               shared_mem=smem)

    return valid[0]


def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):

    n_classes = len(classes)

    classes = cp.array(classes)

    is_binary = True if n_classes == 1 and cp.unique(y) == 2 else False

    sorted_classes = cp.array(classes, dtype=cp.int32)

    col_ind = cp.array(y).copy().astype(cp.int32)

    if not _validate_labels(col_ind, classes):
        raise ValueError("Unseen classes encountered in input")

    row_ind = cp.arange(0, col_ind.shape[0], 1, dtype=cp.int32)

    smem = 4 * sorted_classes.shape[0]
    map_kernel((col_ind.shape[0] / 32,), (32, ),
               (col_ind, col_ind.shape[0], sorted_classes, sorted_classes.shape[0]),
               shared_mem=smem)

    val = cp.full(row_ind.shape[0], pos_label, dtype=cp.int32)

    sp = cp.sparse.coo_matrix((val, (row_ind, col_ind)),
                              shape=(col_ind.shape[0],
                                     sorted_classes.shape[0]),
                              dtype=cp.float32)

    if sparse_output:
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
            raise ValueError("Sparse binarization is only supported with non-zero"
                             "pos_label and zero neg_label, got pos_label=%s and neg_label=%s"
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
        y_mapped = cp.argmax(y.astype(cp.int32), axis=1).astype(cp.int32)

        if not _validate_labels(y_mapped, self.classes_):
            raise ValueError("Unseen classes encountered in input")

        smem = 4 * self.classes_.shape[0]
        inverse_map_kernel((y_mapped.shape[0] / 32,), (32,),
                           (self.classes_, self.classes_.shape[0],
                            y_mapped, y_mapped.shape[0]),
                           shared_mem=smem)

        return y_mapped

    def __getstate__(self):
        state = self.__dict__.copy()
        state['classes_'] = cudf.Series(numba.cuda.to_device(self.classes_))
        return state

    def __setstate__(self, state):
        state['classes_'] = cp.asarray(state["classes_"].to_gpu_array())
        self.__dict__.update(state)



