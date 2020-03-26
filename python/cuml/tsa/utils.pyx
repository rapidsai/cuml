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

import ctypes
import numpy as np
from libc.stdint cimport uintptr_t
from libcpp cimport bool

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.handle cimport cumlHandle
from cuml.utils.input_utils import input_to_cuml_array


cdef extern from "cuml/tsa/divide_batch.h" namespace "ML":
    int divide_batch_build_index(const cumlHandle& handle, const bool* mask,
                                 int* index, int batch_size, int n_obs)

    void divide_batch_execute(const cumlHandle& handle, const float* d_in,
                              const bool* mask, const int* index,
                              float* d_out0, float* d_out1, int batch_size,
                              int n_obs)
    void divide_batch_execute(const cumlHandle& handle, const double* d_in,
                              const bool* mask, const int* index,
                              double* d_out0, double* d_out1, int batch_size,
                              int n_obs)


def divide_batch(original, mask, handle=None):
    """TODO: docs
    (note: takes only cuML arrays as arguments)
    """
    dtype = original.dtype
    n_obs = original.shape[0]
    batch_size = original.shape[1] if len(original.shape) > 1 else 1

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    index = cumlArray.empty(batch_size, np.int32)
    cdef uintptr_t d_index = index.ptr
    cdef uintptr_t d_mask = mask.ptr

    if dtype == np.float32:
        nb_true = divide_batch_build_index(handle_[0],
                                           <bool*> d_mask,
                                           <int*> d_index,
                                           <int> batch_size,
                                           <int> n_obs)
    else:
        nb_true = divide_batch_build_index(handle_[0],
                                           <bool*> d_mask,
                                           <int*> d_index,
                                           <int> batch_size,
                                           <int> n_obs)

    out0 = cumlArray.empty((n_obs, batch_size - nb_true), dtype)
    out1 = cumlArray.empty((n_obs, nb_true), dtype)

    cdef uintptr_t d_out0 = out0.ptr
    cdef uintptr_t d_out1 = out1.ptr
    cdef uintptr_t d_original = original.ptr

    if dtype == np.float32:
        divide_batch_execute(handle_[0],
                             <float*> d_original,
                             <bool*> d_mask,
                             <int*> d_index,
                             <float*> d_out0,
                             <float*> d_out1,
                             <int> batch_size,
                             <int> n_obs)
    else:
        divide_batch_execute(handle_[0],
                             <double*> d_original,
                             <bool*> d_mask,
                             <int*> d_index,
                             <double*> d_out0,
                             <double*> d_out1,
                             <int> batch_size,
                             <int> n_obs)

    return out0, out1
