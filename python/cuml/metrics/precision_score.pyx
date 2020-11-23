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

# distutils: language = c++

import cupy as cp

import cuml.internals
from cuml.raft.common.handle cimport handle_t
from libc.stdint cimport uintptr_t

from cuml.metrics.utils import sorted_unique_labels
from cuml.prims.label import make_monotonic
from cuml.common import input_to_cuml_array
from cuml.raft.common.handle import Handle


cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics":
	double precision_score(const handle_t &handle,
						   const int *y,
						   const int *y_hat,
						   const int n) except +


@cuml.internals.api_return_any()
def cython_precision_score(labels_true, labels_pred, handle=None) -> float:
	"""
	Compute the Precision Score

	Parameters
	----------
	handle : cuml.Handle
	labels_pred : array-like (device or host) shape = (n_samples,)
	    Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
	    ndarray, cuda array interface compliant array like CuPy
	labels_true : array-like (device or host) shape = (n_samples,)
	    Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
	    ndarray, cuda array interface compliant array like CuPy

	Returns
	-------
	float
	  Precision Score, a non-negative value
	"""
	handle = Handle() if handle is None else handle
	cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

	y_true, n_rows, _, dtype = input_to_cuml_array(
	    labels_true,
	    check_dtype=[cp.int32, cp.int64],
	    check_cols=1,
	    deepcopy=True  # deepcopy because we call make_monotonic inplace below
	)

	y_pred, _, _, _ = input_to_cuml_array(
	    labels_pred,
	    check_dtype=dtype,
	    check_rows=n_rows,
	    check_cols=1,
	    deepcopy=True  # deepcopy because we call make_monotonic inplace below
	)

	classes = sorted_unique_labels(y_true, y_pred)

	make_monotonic(y_true, classes=classes, copy=False)
	make_monotonic(y_pred, classes=classes, copy=False)

	cdef uintptr_t ground_truth_ptr = y_true.ptr
	cdef uintptr_t preds_ptr = y_pred.ptr

	lower_class_range = 0
	upper_class_range = len(classes) - 1

	# ERROR CHECKING HERE: only binary is currently supported
	if upper_class_range - lower_class_range + 1 > 2:
		raise ValueError("Only binary labels are currently supported.")

	precision = precision_score(handle_[0],
								<int*> ground_truth_ptr,
								<int*> preds_ptr,
								<int> n_rows)

	return precision

