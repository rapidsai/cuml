#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import typing

import cupy as cp

import cuml.internals
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cupy_array

map_kernel_str = r"""
({0} *x, int x_n, {0} *labels, int n_labels) {

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ {0} label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x)
    label_cache[i] = labels[i];

  if(tid >= x_n) return;
  __syncthreads();

  {0} unmapped_label = x[tid];
  for(int i = 0; i < n_labels; i++) {
    if(label_cache[i] == unmapped_label) {
      x[tid] = i;
      return;
    }
  }
  x[tid] = n_labels+1;
}
"""


validate_kernel_str = r"""
({0} *x, int x_n, {0} *labels, int n_labels, int *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ {0} label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x)
    label_cache[i] = labels[i];

  if(tid >= x_n) return;

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
"""


inverse_map_kernel_str = r"""
({0} *labels, int n_labels, {0} *x, int x_n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ {0} label_cache[];
  for(int i = threadIdx.x; i < n_labels; i+=blockDim.x) {
    label_cache[i] = labels[i];
  }

  if(tid >= x_n) return;
  __syncthreads();

  {0} mapped_label = x[tid];
  {0} original_label = label_cache[mapped_label];

  x[tid] = original_label;
}
"""


def _map_kernel(dtype):
    return cuda_kernel_factory(map_kernel_str, (dtype,), "map_labels_kernel")


def _inverse_map_kernel(dtype):
    return cuda_kernel_factory(
        inverse_map_kernel_str, (dtype,), "inv_map_labels_kernel"
    )


def _validate_kernel(dtype):
    return cuda_kernel_factory(
        validate_kernel_str, (dtype,), "validate_labels_kernel"
    )


@cuml.internals.api_return_generic(input_arg="labels", get_output_type=True)
def make_monotonic(
    labels, classes=None, copy=False
) -> typing.Tuple[CumlArray, CumlArray]:
    """
    Takes a set of labels that might not be drawn from the
    set [0, n-1] and renumbers them to be drawn that
    interval.

    Replaces labels not present in classes by len(classes)+1.

    Parameters
    ----------

    labels : array-like of size (n,) labels to convert
    classes : array-like of size (n_classes,) the unique
              set of classes in the set of labels
    copy : boolean if true, a copy will be returned and the
           operation will not be done in place.

    Returns
    -------

    mapped_labels : array-like of size (n,)
    classes : array-like of size (n_classes,)
    """
    labels = input_to_cupy_array(labels, deepcopy=copy).array

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    if classes is None:
        classes = cp.unique(labels)
    else:
        classes = input_to_cupy_array(classes).array

    smem = labels.dtype.itemsize * int(classes.shape[0])

    map_labels = _map_kernel(labels.dtype)
    map_labels(
        (math.ceil(labels.shape[0] / 32),),
        (32,),
        (labels, labels.shape[0], classes, classes.shape[0]),
        shared_mem=smem,
    )

    return labels, classes


@cuml.internals.api_return_any()
def check_labels(labels, classes) -> bool:
    """
    Validates that a set of labels is drawn from the unique
    set of given classes.

    Parameters
    ----------

    labels : array-like of size (n,) labels to validate
    classes : array-like of size (n_classes,) the unique
              set of classes to verify

    Returns
    -------

    result : boolean
    """

    labels = input_to_cupy_array(labels, order="K").array
    classes = input_to_cupy_array(classes, order="K").array

    if labels.dtype != classes.dtype:
        raise ValueError(
            "Labels and classes must have same dtype (%s != %s"
            % (labels.dtype, classes.dtype)
        )

    if labels.ndim != 1:
        raise ValueError("Labels array must be 1D")

    valid = cp.array([1])

    smem = labels.dtype.itemsize * int(classes.shape[0])
    validate = _validate_kernel(labels.dtype)
    validate(
        (math.ceil(labels.shape[0] / 32),),
        (32,),
        (labels, labels.shape[0], classes, classes.shape[0], valid),
        shared_mem=smem,
    )

    return valid[0] == 1


@cuml.internals.api_return_array(input_arg="labels", get_output_type=True)
def invert_labels(labels, classes, copy=False) -> CumlArray:
    """
    Takes a set of labels that have been mapped to be drawn
    from a monotonically increasing set and inverts them to
    back to the original set of classes.

    Parameters
    ----------

    labels : array-like of size (n,) labels to invert
    classes : array-like of size (n_classes,) the unique set
              of classes for inversion. It is assumed that
              the classes are ordered by their corresponding
              monotonically increasing label.
    copy : boolean if true, a copy will be returned and the
           operation will not be done in place.

    Returns
    -------

    inverted labels : array-like of size (n,)

    """
    labels = input_to_cupy_array(labels, deepcopy=copy).array
    classes = input_to_cupy_array(classes).array

    if labels.dtype != classes.dtype:
        raise ValueError(
            "Labels and classes must have same dtype (%s != %s"
            % (labels.dtype, classes.dtype)
        )
    smem = labels.dtype.itemsize * len(classes)
    inverse_map = _inverse_map_kernel(labels.dtype)
    inverse_map(
        (math.ceil(len(labels) / 32),),
        (32,),
        (classes, len(classes), labels, len(labels)),
        shared_mem=smem,
    )

    return labels
