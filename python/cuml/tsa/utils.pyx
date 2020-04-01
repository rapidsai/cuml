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
from libcpp.vector cimport vector

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.handle cimport cumlHandle
from cuml.utils.input_utils import input_to_cuml_array


cdef extern from "cuml/tsa/ts_utils.h" namespace "ML":
    int divide_by_mask_build_index(const cumlHandle& handle, const bool* mask,
                                   int* index, int batch_size)

    void divide_by_mask_execute(const cumlHandle& handle, const float* d_in,
                                const bool* mask, const int* index,
                                float* d_out0, float* d_out1, int batch_size,
                                int n_obs)
    void divide_by_mask_execute(const cumlHandle& handle, const double* d_in,
                                const bool* mask, const int* index,
                                double* d_out0, double* d_out1,
                                int batch_size, int n_obs)
    void divide_by_mask_execute(const cumlHandle& handle, const int* d_in,
                                const bool* mask, const int* index,
                                int* d_out0, int* d_out1, int batch_size,
                                int n_obs)

    void divide_by_min_build_index(const cumlHandle& handle,
                                   const float* d_matrix, int* d_batch,
                                   int* d_index, int* h_size,
                                   int batch_size, int n_sub)
    void divide_by_min_build_index(const cumlHandle& handle,
                                   const double* d_matrix, int* d_batch,
                                   int* d_index, int* h_size,
                                   int batch_size, int n_sub)

    void divide_by_min_execute(const cumlHandle& handle, const float* d_in,
                               const int* d_batch, const int* d_index,
                               float** hd_out, int batch_size, int n_sub,
                               int n_obs)
    void divide_by_min_execute(const cumlHandle& handle, const double* d_in,
                               const int* d_batch, const int* d_index,
                               double** hd_out, int batch_size, int n_sub,
                               int n_obs)
    void divide_by_min_execute(const cumlHandle& handle, const int* d_in,
                               const int* d_batch, const int* d_index,
                               int** hd_out, int batch_size, int n_sub,
                               int n_obs)

# TODO: tests?


def divide_by_mask(original, mask, batch_id=None, handle=None):
    """TODO: docs
    (note: takes only cuML arrays as arguments)
    (note: when a sub-batch has size zero, it is set to None and the other
     to the original batch, not a copy!)
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

    # Compute the index of each series in their new batch
    nb_true = divide_by_mask_build_index(handle_[0],
                                         <bool*> d_mask,
                                         <int*> d_index,
                                         <int> batch_size)

    out0 = cumlArray.empty((n_obs, batch_size - nb_true), dtype)
    out1 = cumlArray.empty((n_obs, nb_true), dtype)

    # Type declarations (can't be in if-else statements)
    cdef uintptr_t d_out0
    cdef uintptr_t d_out1
    cdef uintptr_t d_original = original.ptr
    cdef uintptr_t d_batch0_id
    cdef uintptr_t d_batch1_id
    cdef uintptr_t d_batch_id

    # If the sub-batch 1 is empty
    if nb_true == 0:
        out0 = original
        out1 = None
        if batch_id is not None:
            batch0_id = batch_id
            batch1_id = None

    # If the sub-batch 0 is empty
    elif nb_true == batch_size:
        out0 = None
        out1 = original
        if batch_id is not None:
            batch0_id = None
            batch1_id = batch_id

    # If both sub-batches have elements
    else:
        out0 = cumlArray.empty((n_obs, batch_size - nb_true), dtype)
        out1 = cumlArray.empty((n_obs, nb_true), dtype)
        d_out0 = out0.ptr
        d_out1 = out1.ptr

        # Build the two sub-batches
        if dtype == np.float32:
            divide_by_mask_execute(handle_[0],
                                   <float*> d_original,
                                   <bool*> d_mask,
                                   <int*> d_index,
                                   <float*> d_out0,
                                   <float*> d_out1,
                                   <int> batch_size,
                                   <int> n_obs)
        else:
            divide_by_mask_execute(handle_[0],
                                  <double*> d_original,
                                  <bool*> d_mask,
                                  <int*> d_index,
                                  <double*> d_out0,
                                  <double*> d_out1,
                                  <int> batch_size,
                                  <int> n_obs)

        # Also keep track of the original id of the series in the batch
        if batch_id is not None:
            # TODO: check int dtype! Convert to np.int32 if needed

            batch0_id = cumlArray.empty(batch_size - nb_true, np.int32)
            batch1_id = cumlArray.empty(nb_true, np.int32)
            d_batch0_id = batch0_id.ptr
            d_batch1_id = batch1_id.ptr
            d_batch_id = batch_id.ptr

            divide_by_mask_execute(handle_[0],
                                   <int*> d_batch_id,
                                   <bool*> d_mask,
                                   <int*> d_index,
                                   <int*> d_batch0_id,
                                   <int*> d_batch1_id,
                                   <int> batch_size,
                                   <int> 1)
    
    # Return two values or two tuples depending on the function args
    if batch_id is not None:
        return (out0, batch0_id), (out1, batch1_id)
    else:
        return out0, out1


def divide_by_min(original, metrics, batch_id=None, handle=None):
    """TODO: docs
    (note: takes only cuML arrays as arguments)
    (note: when a sub-batch has size zero, it is set to None)
    """
    dtype = original.dtype
    n_obs = original.shape[0]
    n_sub = metrics.shape[1]
    batch_size = original.shape[1] if len(original.shape) > 1 else 1

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    batch_buffer = cumlArray.empty(batch_size, np.int32)
    index_buffer = cumlArray.empty(batch_size, np.int32)
    cdef vector[int] size_buffer
    size_buffer.resize(n_sub)

    cdef uintptr_t d_metrics = metrics.ptr
    cdef uintptr_t d_batch = batch_buffer.ptr
    cdef uintptr_t d_index = index_buffer.ptr

    # Compute which sub-batch each series belongs to, its position in
    # the sub-batch, and the size of each sub-batch
    if dtype == np.float32:
        divide_by_min_build_index(handle_[0],
                                  <float*> d_metrics,
                                  <int*> d_batch,
                                  <int*> d_index,
                                  <int*> size_buffer.data(),
                                  <int> batch_size,
                                  <int> n_sub)
    else:
        divide_by_min_build_index(handle_[0],
                                  <double*> d_metrics,
                                  <int*> d_batch,
                                  <int*> d_index,
                                  <int*> size_buffer.data(),
                                  <int> batch_size,
                                  <int> n_sub)

    # Build a list of cuML arrays for the sub-batches and a vector of pointers
    # to be passed to the next C++ step
    sub_batches = [cumlArray.empty((n_obs, s), dtype) if s else None
                   for s in size_buffer]
    cdef vector[uintptr_t] sub_ptr
    sub_ptr.resize(n_sub)
    for i in range(n_sub):
        if size_buffer[i]:
            sub_ptr[i] = <uintptr_t> sub_batches[i].ptr
        else:
            sub_ptr[i] = <uintptr_t> NULL

    # Execute the batch sub-division
    cdef uintptr_t d_original = original.ptr
    if dtype == np.float32:
        divide_by_min_execute(handle_[0],
                              <float*> d_original,
                              <int*> d_batch,
                              <int*> d_index,
                              <float**> sub_ptr.data(),
                              <int> batch_size,
                              <int> n_sub,
                              <int> n_obs)
    else:
        divide_by_min_execute(handle_[0],
                              <double*> d_original,
                              <int*> d_batch,
                              <int*> d_index,
                              <double**> sub_ptr.data(),
                              <int> batch_size,
                              <int> n_sub,
                              <int> n_obs)

    # Keep track of the id of the series if requested
    cdef vector[uintptr_t] id_ptr
    cdef uintptr_t d_batch_id
    if batch_id is not None:
        sub_id = [cumlArray.empty(s, np.int32) if s else None
                  for s in size_buffer]
        id_ptr.resize(n_sub)
        for i in range(n_sub):
            if size_buffer[i]:
                id_ptr[i] = <uintptr_t> sub_id[i].ptr
            else:
                id_ptr[i] = <uintptr_t> NULL

        # TODO: check int dtype! Convert to np.int32 if needed
        
        d_batch_id = batch_id.ptr
        divide_by_min_execute(handle_[0],
                              <int*> d_batch_id,
                              <int*> d_batch,
                              <int*> d_index,
                              <int**> id_ptr.data(),
                              <int> batch_size,
                              <int> n_sub,
                              <int> 1)

    # Return the sub-batches and optionally the id of the series
    if batch_id is not None:
        return sub_batches, sub_id
    else:
        return sub_batches
