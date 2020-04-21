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

import cupy as cp

import cuml
from cuml.common.array import CumlArray as cumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.tsa.arima import ARIMA
from cuml.tsa.seasonality import seas_test
from cuml.tsa.stationarity import kpss_test
from cuml.utils.input_utils import input_to_cuml_array


# TODO:
# - change interface to match the new fable package instead of deprecated
#    forecast package?
#    -> use int/range/sequence syntax for params?
# - truncate argument to use only last values with CSS
# - Box-Cox transformations? (parameter lambda)
# - summary method with recap of the models used
# - integrate cuML logging system
# - unit tests
# - use output_type as soon as cuML array change in ARIMA is merged


cdef extern from "cuml/tsa/auto_arima.h" namespace "ML":
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

    void cpp_build_division_map "ML::build_division_map" (
        const cumlHandle& handle, const int* const* hd_id, const int* h_size,
        int* d_id_to_pos, int* d_id_to_model, int batch_size, int n_sub)

    void cpp_merge_series "ML::merge_series" (
        const cumlHandle& handle, const float* const* hd_in,
        const int* d_id_to_pos, const int* d_id_to_sub, float* d_out,
        int batch_size, int n_sub, int n_obs)
    void cpp_merge_series "ML::merge_series" (
        const cumlHandle& handle, const double* const* hd_in,
        const int* d_id_to_pos, const int* d_id_to_sub, double* d_out,
        int batch_size, int n_sub, int n_obs)

tests_map = {
    "kpss": kpss_test,
    "seas": seas_test,
}

class AutoARIMA(Base):
    r"""TODO: docs
    """
    
    def __init__(self, y,
                 handle=None,
                 verbose=0,
                 output_type=None):
        super().__init__(handle, verbose, output_type)

        # Get device array. Float64 only for now.
        self.d_y, self.n_obs, self.batch_size, self.dtype \
            = input_to_cuml_array(y, check_dtype=np.float64)

    def fit(self,
            s=None,
            d=None,
            D=None,
            max_d=2,
            max_D=1, # TODO: remove if we never use D=2
            start_p=2, # TODO: start at 0?
            start_q=2,
            start_P=1,
            start_Q=1,
            max_p=4, # TODO: support p=5 / q=5 in ARIMA
            max_q=4,
            max_P=2,
            max_Q=2,
            ic="aicc", # TODO: which one to use by default?
            test="kpss",
            seasonal_test="seas",
            search_method="auto",
            final_method="ml"):
        """TODO: docs
        """
        # Notes:
        #  - We iteratively divide the dataset as we decide parameters, so
        #    it's important to make sure that we don't keep the unused arrays
        #    alive, so they can get garbage collected.
        #  - As we divide the dataset, we also keep track of the original
        #    index of each series in the batch, to construct the final map at
        #    the end.

        # Parse input parameters
        ic = ic.lower()
        test = test.lower()
        seasonal_test = seasonal_test.lower()
        if s == 1:  # R users might use s=1 for a non-seasonal dataset
            s = None
        if search_method == "auto":
            search_method = "css" if self.n_obs >= 100 and s >= 4 else "ml"

        # Box-Cox transform
        # TODO: handle it

        # Original index
        d_index, *_ = input_to_cuml_array(np.r_[:self.batch_size],
                                          convert_to_dtype=np.int32)
        # TODO: worth building on GPU?

        #
        # Choose the hyper-parameter D
        #
        if self.verbose:
            print("Deciding D...")
        if not s:
            # Non-seasonal -> D=0
            data_D = {0: (self.d_y, d_index)}
        elif D is not None:
            # D is specified by the user
            data_D = {D: (self.d_y, d_index)}
        else:
            # D is chosen with a seasonal differencing test
            if seasonal_test not in tests_map:
                raise ValueError("Unknown seasonal diff test: {}"
                                 .format(seasonal_test))
            mask = tests_map[seasonal_test](self.d_y, s)
            data_D = {}
            (out0, index0), (out1, index1) = _divide_by_mask(self.d_y, mask,
                                                             d_index)
            if out0 is not None:
                data_D[0] = (out0, index0)
            if out1 is not None:
                data_D[1] = (out1, index1)
            del mask, out0, index0, out1, index1
        # TODO: can D be 2?

        #
        # Choose the hyper-parameter d
        #
        if self.verbose:
            print("Deciding d...")
        data_dD = {}
        for D_ in data_D:
            if d is not None:
                # d is specified by the user
                data_dD[(d, D_)] = data_D[D_]
            else:
                # d is decided with stationarity tests
                if test not in tests_map:
                    raise ValueError("Unknown stationarity test: {}"
                                     .format(test))
                data_temp, id_temp = data_D[D_]
                for d_ in range(min(max_d, 2 - D_)):
                    mask = tests_map[test](data_temp, d_, D_, s)
                    (out0, index0), (out1, index1) \
                        = _divide_by_mask(data_temp, mask, id_temp)
                    if out1 is not None:
                        data_dD[(d_, D_)] = (out1, index1)
                    if out0 is not None:
                        (data_temp, id_temp) = (out0, index0)
                    else:
                        break
                else: # (when the for loop reaches its end naturally)
                    # The remaining series are assigned the max possible d
                    data_dD[(min(max_d, 2 - D_), D_)] = (data_temp, id_temp)
                del data_temp, id_temp, mask, out0, index0, out1, index1
        del data_D

        # Limit the number of parameters to what we can handle
        # TODO: handle more than 4 in the Jones transform
        max_p = min(max_p, 4)
        max_q = min(max_q, 4)
        if s:
            max_p = min(max_p, s - 1)
            max_q = min(max_q, s - 1)
        max_P = min(max_P, 4) if s else 0
        max_Q = min(max_Q, 4) if s else 0
        start_p = min(start_p, max_p)
        start_q = min(start_q, max_p)
        start_P = min(start_P, max_p)
        start_Q = min(start_Q, max_p)

        #
        # Choose the hyper-parameters p, q, P, Q, k
        #
        if self.verbose:
            print("Deciding p, q, P, Q, k...")
        # TODO: try nice progress bar when using verbose for grid search
        #       (can use different levels of verbose)
        self.models = []
        id_tracker = []
        for (d_, D_) in data_dD:
            data_temp, id_temp = data_dD[(d_, D_)]
            batch_size = data_temp.shape[1] if len(data_temp.shape) > 1 else 1
            k_ = 1 if d_ + D_ <= 1 else 0

            # Grid search
            # TODO: think about a (partially) step-wise parallel approach
            all_ic = []
            all_orders = []
            for p_ in range(start_p, max_p + 1):
                for q_ in range(start_q, max_q + 1):
                    for P_ in range(start_P, max_P + 1):
                        for Q_ in range(start_Q, max_Q + 1):
                            if p_ + q_ + P_ + Q_ + k_ == 0:
                                continue
                            s_ = s if (P_ + D_ + Q_) else 0
                            # TODO: raise issue that input_to_cuml_array
                            #       should support cuML arrays
                            model = ARIMA(cp.asarray(data_temp), (p_, d_, q_),
                                          (P_, D_, Q_, s_), k_, self.handle)
                            if self.verbose:
                                print(" -", str(model))
                            model.fit(method=search_method)
                            all_ic.append(model._ic(ic))
                            all_orders.append((p_, q_, P_, Q_, s_, k_))
                            del model

            # Organize the results into a matrix
            n_models = len(all_orders)
            ic_matrix, *_ = input_to_cuml_array(
                cp.concatenate([cp.asarray(ic_arr).reshape(batch_size, 1)
                                for ic_arr in all_ic], 1))

            # Divide the batch, choosing the best model for each series
            sub_batches, sub_id = _divide_by_min(data_temp, ic_matrix, id_temp)
            for i in range(n_models):
                if sub_batches[i] is None:
                    continue
                p_, q_, P_, Q_, s_, k_ = all_orders[i]
                self.models.append(ARIMA(cp.asarray(sub_batches[i]),
                                         order=(p_, d_, q_),
                                         seasonal_order=(P_, D_, Q_, s_),
                                         fit_intercept=k_,
                                         handle=self.handle))
                id_tracker.append(sub_id[i])

            del all_ic, all_orders, ic_matrix, sub_batches, sub_id

        # TODO: try different k_ on the best model?

        if self.verbose:
            print("Fitting final models...")
        for model in self.models:
            if self.verbose:
                print(" - {}".format(model))
            model.fit(method=final_method)

        # Build a map to match each series to its model and position in the
        # sub-batch
        if self.verbose:
            print("Finalizing...")
        self.id_to_model, self.id_to_pos = _build_division_map(id_tracker,
                                                               self.batch_size)

    def predict(self, start=0, end=None):
        """TODO: docs
        """
        # Compute predictions for each model
        predictions = []
        for model in self.models:
            pred, *_ = input_to_cuml_array(model.predict(start, end))
            # TODO: no need for cast after cuML array PR is merged
            predictions.append(pred)
        
        # Put all the predictions together
        return _merge_series(predictions, self.id_to_model, self.id_to_pos,
                             self.batch_size)

    def forecast(self, nsteps):
        """TODO: docs
        """
        return self.predict(self.n_obs, self.n_obs + nsteps)


# Helper functions

def _divide_by_mask(original, mask, batch_id, handle=None):
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
        batch0_id = batch_id
        batch1_id = None

    # If the sub-batch 0 is empty
    elif nb_true == batch_size:
        out0 = None
        out1 = original
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

    return (out0, batch0_id), (out1, batch1_id)



def _divide_by_min(original, metrics, batch_id, handle=None):
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

    return sub_batches, sub_id


def _build_division_map(id_tracker, batch_size, handle=None):
    """TODO: docs
    """
    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    n_sub = len(id_tracker)

    id_to_pos = cumlArray.empty(batch_size, np.int32)
    id_to_model = cumlArray.empty(batch_size, np.int32)

    cdef vector[uintptr_t] id_ptr
    cdef vector[int] size_vec
    id_ptr.resize(n_sub)
    size_vec.resize(n_sub)
    for i in range(n_sub):
        id_ptr[i] = id_tracker[i].ptr
        size_vec[i] = len(id_tracker[i])

    cdef uintptr_t hd_id = <uintptr_t> id_ptr.data()
    cdef uintptr_t h_size = <uintptr_t> size_vec.data()
    cdef uintptr_t d_id_to_pos = id_to_pos.ptr
    cdef uintptr_t d_id_to_model = id_to_model.ptr
    
    cpp_build_division_map(handle_[0],
                           <const int**> hd_id,
                           <int*> h_size,
                           <int*> d_id_to_pos,
                           <int*> d_id_to_model,
                           <int> batch_size,
                           <int> n_sub)

    return id_to_model, id_to_pos


def _merge_series(data_in, id_to_sub, id_to_pos, batch_size, handle=None):
    """TODO: docs
    """
    dtype = data_in[0].dtype
    n_obs = data_in[0].shape[0]
    n_sub = len(data_in)

    if handle is None:
        handle = cuml.common.handle.Handle()
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef vector[uintptr_t] in_ptr
    in_ptr.resize(n_sub)
    for i in range(n_sub):
        in_ptr[i] = data_in[i].ptr

    data_out = cumlArray.empty((n_obs, batch_size), dtype)

    cdef uintptr_t hd_in = <uintptr_t> in_ptr.data()
    cdef uintptr_t d_id_to_pos = id_to_pos.ptr
    cdef uintptr_t d_id_to_sub = id_to_sub.ptr
    cdef uintptr_t d_out = data_out.ptr

    if dtype == np.float32:
        cpp_merge_series(handle_[0],
                         <const float**> hd_in,
                         <int*> d_id_to_pos,
                         <int*> d_id_to_sub,
                         <float*> d_out,
                         <int> batch_size,
                         <int> n_sub,
                         <int> n_obs)
    else:
        cpp_merge_series(handle_[0],
                         <const double**> hd_in,
                         <int*> d_id_to_pos,
                         <int*> d_id_to_sub,
                         <double*> d_out,
                         <int> batch_size,
                         <int> n_sub,
                         <int> n_obs)

    return data_out

