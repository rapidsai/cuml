#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

from cuml.cluster import DBSCAN

import ctypes
import numpy as np
import cupy as cp

from libcpp cimport bool
from libc.stdint cimport uintptr_t, int64_t

from cuml.common.array import CumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common import using_output_type


cdef extern from "cuml/cluster/dbscan.hpp" \
        namespace "ML::Dbscan":

    cdef void fit(handle_t& handle,
                  float *input,
                  int n_rows,
                  int n_cols,
                  float eps,
                  int min_pts,
                  int *labels,
                  int *core_sample_indices,
                  size_t max_mbytes_per_batch,
                  int verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  double *input,
                  int n_rows,
                  int n_cols,
                  double eps,
                  int min_pts,
                  int *labels,
                  int *core_sample_indices,
                  size_t max_mbytes_per_batch,
                  int verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  float *input,
                  int64_t n_rows,
                  int64_t n_cols,
                  double eps,
                  int min_pts,
                  int64_t *labels,
                  int64_t *core_sample_indices,
                  size_t max_mbytes_per_batch,
                  int verbosity,
                  bool opg) except +

    cdef void fit(handle_t& handle,
                  double *input,
                  int64_t n_rows,
                  int64_t n_cols,
                  double eps,
                  int min_pts,
                  int64_t *labels,
                  int64_t *core_sample_indices,
                  size_t max_mbytes_per_batch,
                  int verbosity,
                  bool opg) except +


class DBSCANMG(DBSCAN):
    """
    A Multi-Node Multi-GPU implementation of DBSCAN

    NOTE: This implementation of DBSCAN is meant to be used with an
    initialized cumlCommunicator instance inside an existing distributed
    system. Refer to the Dask BSCAN implementation in
    `cuml.dask.cluster.dbscan`.
    """

    def __init__(self, **kwargs):
        super(DBSCANMG, self).__init__(**kwargs)

    @generate_docstring(skip_parameters_heading=True)
    def fit(self, X, out_dtype="int32"):
        """
        Perform DBSCAN clustering in a multi-node multi-GPU setting.

        Parameters
        ----------
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.
        """
        self._set_base_attributes(n_features=X)

        if out_dtype not in ["int32", np.int32, "int64", np.int64]:
            raise ValueError("Invalid value for out_dtype. "
                             "Valid values are {'int32', 'int64', "
                             "np.int32, np.int64}")

        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_dtype=[np.float32, np.float64])

        cdef uintptr_t input_ptr = X_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        self.labels_ = CumlArray.empty(n_rows, dtype=out_dtype)
        cdef uintptr_t labels_ptr = self.labels_.ptr

        cdef uintptr_t core_sample_indices_ptr = <uintptr_t> NULL

        # Create the output core_sample_indices only if needed
        if self.calc_core_sample_indices:
            self.core_sample_indices_ = \
                CumlArray.empty(n_rows, dtype=out_dtype)
            core_sample_indices_ptr = self.core_sample_indices_.ptr

        if self.dtype == np.float32:
            if out_dtype == "int32" or out_dtype is np.int32:
                fit(handle_[0],
                    <float*>input_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <float> self.eps,
                    <int> self.min_samples,
                    <int*> labels_ptr,
                    <int*> core_sample_indices_ptr,
                    <size_t>self.max_mbytes_per_batch,
                    <int> self.verbose,
                    <bool> True)
            else:
                fit(handle_[0],
                    <float*>input_ptr,
                    <int64_t> n_rows,
                    <int64_t> n_cols,
                    <float> self.eps,
                    <int> self.min_samples,
                    <int64_t*> labels_ptr,
                    <int64_t*> core_sample_indices_ptr,
                    <size_t>self.max_mbytes_per_batch,
                    <int> self.verbose,
                    <bool> True)

        else:
            if out_dtype == "int32" or out_dtype is np.int32:
                fit(handle_[0],
                    <double*>input_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <double> self.eps,
                    <int> self.min_samples,
                    <int*> labels_ptr,
                    <int*> core_sample_indices_ptr,
                    <size_t> self.max_mbytes_per_batch,
                    <int> self.verbose,
                    <bool> True)
            else:
                fit(handle_[0],
                    <double*>input_ptr,
                    <int64_t> n_rows,
                    <int64_t> n_cols,
                    <double> self.eps,
                    <int> self.min_samples,
                    <int64_t*> labels_ptr,
                    <int64_t*> core_sample_indices_ptr,
                    <size_t> self.max_mbytes_per_batch,
                    <int> self.verbose,
                    <bool> True)

        # make sure that the `dbscanFit` is complete before the following
        # delete call happens
        self.handle.sync()
        del(X_m)

        # Finally, resize the core_sample_indices array if necessary
        if self.calc_core_sample_indices:

            # Temp convert to cupy array (better than using `cupy.asarray`)
            with using_output_type("cupy"):

                # First get the min index. These have to monotonically
                # increasing, so the min index should be the first returned -1
                min_index = cp.argmin(self.core_sample_indices_).item()

                # Check for the case where there are no -1's
                if ((min_index == 0 and
                     self.core_sample_indices_[min_index].item() != -1)):
                    # Nothing to delete. The array has no -1's
                    pass
                else:
                    self.core_sample_indices_ = \
                        self.core_sample_indices_[:min_index]

        return self
