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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cudf
import ctypes
import numpy as np
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t

import cuml
from cuml.common.handle cimport cumlHandle
from cuml.utils.input_utils import input_to_dev_array


cdef extern from "cuml/tsa/stationarity.h" namespace "ML":
  int cpp_stationarity "ML::Stationarity::stationarity" (
      const cumlHandle& handle,
      const double* y_d,
      int* d,
      int n_batches,
      int n_samples,
      double pval_threshold)


def stationarity(y, pval_threshold=0.05, handle=None):
    """
    TODO: write docs
    """
    # TODO: don't impose dtype?
    cdef uintptr_t y_d_ptr
    y_d, y_d_ptr, n_samples, n_batches, dtype \
        = input_to_dev_array(y, check_dtype=np.float64)

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef vector[int] d
    d.resize(n_batches)

    # Call C++ function
    ret_value = cpp_stationarity(handle_[0], <double*> y_d_ptr, <int*> d.data(),
                                 <int> n_batches, <int> n_samples,
                                 <double> pval_threshold)

    if ret_value < 0:
        raise ValueError("Stationarity test failed for d=0 or 1.")
    
    return d
