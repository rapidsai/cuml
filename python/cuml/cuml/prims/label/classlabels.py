#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp

from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.internals.input_utils import input_to_cupy_array


def _get_max_shared_memory_per_block():
    """Get the maximum shared memory per block for the current device."""
    device = cp.cuda.Device()
    # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
    return device.attributes.get("MaxSharedMemoryPerBlock", 49152)


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


def _make_monotonic_fallback(labels, classes):
    """
    Fallback implementation using CuPy when shared memory is insufficient.
    Maps labels to monotonic indices [0, n_classes-1].
    Labels not in classes are mapped to n_classes+1.
    """
    # Create a mapping using searchsorted on sorted classes
    sorted_indices = cp.argsort(classes)
    sorted_classes = classes[sorted_indices]

    # Find where each label would be inserted in sorted classes
    insert_positions = cp.searchsorted(sorted_classes, labels)

    # Check if the labels actually match the classes at those positions
    # Clamp positions to valid range for comparison
    clamped_positions = cp.clip(insert_positions, 0, len(classes) - 1)
    matches = sorted_classes[clamped_positions] == labels

    # Map back to original class indices
    mapped_labels = cp.where(
        matches,
        sorted_indices[clamped_positions],
        len(classes) + 1,
    )

    # Copy result back to labels array (in-place modification)
    labels[:] = mapped_labels.astype(labels.dtype)
    return labels, classes


def make_monotonic(labels, classes=None, copy=False):
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
    max_smem = _get_max_shared_memory_per_block()

    # Use fallback if shared memory requirement exceeds device limit
    if smem > max_smem:
        return _make_monotonic_fallback(labels, classes)

    map_labels = _map_kernel(labels.dtype)
    map_labels(
        (math.ceil(labels.shape[0] / 32),),
        (32,),
        (labels, labels.shape[0], classes, classes.shape[0]),
        shared_mem=smem,
    )

    return labels, classes


def _check_labels_fallback(labels, classes) -> bool:
    """
    Fallback implementation using CuPy when shared memory is insufficient.
    """
    return bool(cp.all(cp.isin(labels, classes)))


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

    smem = labels.dtype.itemsize * int(classes.shape[0])
    max_smem = _get_max_shared_memory_per_block()

    # Use fallback if shared memory requirement exceeds device limit
    if smem > max_smem:
        return _check_labels_fallback(labels, classes)

    valid = cp.array([1])

    validate = _validate_kernel(labels.dtype)
    validate(
        (math.ceil(labels.shape[0] / 32),),
        (32,),
        (labels, labels.shape[0], classes, classes.shape[0], valid),
        shared_mem=smem,
    )

    return valid[0] == 1


def _invert_labels_fallback(labels, classes):
    """
    Fallback implementation using CuPy when shared memory is insufficient.
    """
    # Simple indexing: labels contains indices into classes
    inverted = classes[labels]
    labels[:] = inverted
    return labels


def invert_labels(labels, classes, copy=False):
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
    max_smem = _get_max_shared_memory_per_block()

    # Use fallback if shared memory requirement exceeds device limit
    if smem > max_smem:
        return _invert_labels_fallback(labels, classes)

    inverse_map = _inverse_map_kernel(labels.dtype)
    inverse_map(
        (math.ceil(len(labels) / 32),),
        (32,),
        (classes, len(classes), labels, len(labels)),
        shared_mem=smem,
    )

    return labels
