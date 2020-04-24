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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
import cupy as cp
from cuml.metrics.utils import sorted_unique_labels
from cuml.prims.label import make_monotonic
from libc.stdint cimport uintptr_t
from cuml.utils import with_cupy_rmm, input_to_cuml_array


@with_cupy_rmm
def prepare_cluster_metric_inputs(labels_true, labels_pred):
    """Helper function to avoid code duplication for homogeneity score, mutual
    info score and completeness score.
    """
    y_true, n_rows, _, dtype = input_to_cuml_array(
        labels_true,
        check_dtype=[cp.int32, cp.int64],
        check_cols=1
    )

    y_pred, _, _, _ = input_to_cuml_array(
        labels_pred,
        check_dtype=dtype,
        check_rows=n_rows,
        check_cols=1
    )

    cdef uintptr_t preds_ptr = y_pred.ptr
    cdef uintptr_t ground_truth_ptr = y_true.ptr

    classes = sorted_unique_labels(y_true, y_pred)

    # TODO: Do not call make_monotonic inplace and create a new CumlArray
    y_true = make_monotonic(y_true, classes=classes, copy=False)[0]
    y_pred = make_monotonic(y_pred, classes=classes, copy=False)[0]

    lower_class_range = 0
    upper_class_range = len(classes)

    return (ground_truth_ptr, preds_ptr,
            n_rows,
            lower_class_range, upper_class_range)
