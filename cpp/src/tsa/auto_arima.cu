/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "auto_arima.cuh"

#include <cuml/tsa/auto_arima.h>

namespace ML {

int divide_by_mask_build_index(const raft::handle_t& handle, const bool* d_mask,
                               int* d_index, int batch_size) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  return ML::TimeSeries::divide_by_mask_build_index(d_mask, d_index, batch_size,
                                                    allocator, stream);
}

template <typename DataT>
inline void divide_by_mask_execute_helper(const raft::handle_t& handle,
                                          const DataT* d_in, const bool* d_mask,
                                          const int* d_index, DataT* d_out0,
                                          DataT* d_out1, int batch_size,
                                          int n_obs) {
  cudaStream_t stream = handle.get_stream();
  ML::TimeSeries::divide_by_mask_execute(d_in, d_mask, d_index, d_out0, d_out1,
                                         batch_size, n_obs, stream);
}

void divide_by_mask_execute(const raft::handle_t& handle, const float* d_in,
                            const bool* d_mask, const int* d_index,
                            float* d_out0, float* d_out1, int batch_size,
                            int n_obs) {
  divide_by_mask_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                                batch_size, n_obs);
}

void divide_by_mask_execute(const raft::handle_t& handle, const double* d_in,
                            const bool* d_mask, const int* d_index,
                            double* d_out0, double* d_out1, int batch_size,
                            int n_obs) {
  divide_by_mask_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                                batch_size, n_obs);
}

void divide_by_mask_execute(const raft::handle_t& handle, const int* d_in,
                            const bool* d_mask, const int* d_index, int* d_out0,
                            int* d_out1, int batch_size, int n_obs) {
  divide_by_mask_execute_helper(handle, d_in, d_mask, d_index, d_out0, d_out1,
                                batch_size, n_obs);
}

template <typename DataT>
inline void divide_by_min_build_index_helper(const raft::handle_t& handle,
                                             const DataT* d_matrix,
                                             int* d_batch, int* d_index,
                                             int* h_size, int batch_size,
                                             int n_sub) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  ML::TimeSeries::divide_by_min_build_index(
    d_matrix, d_batch, d_index, h_size, batch_size, n_sub, allocator, stream);
}

void divide_by_min_build_index(const raft::handle_t& handle,
                               const float* d_matrix, int* d_batch,
                               int* d_index, int* h_size, int batch_size,
                               int n_sub) {
  divide_by_min_build_index_helper(handle, d_matrix, d_batch, d_index, h_size,
                                   batch_size, n_sub);
}

void divide_by_min_build_index(const raft::handle_t& handle,
                               const double* d_matrix, int* d_batch,
                               int* d_index, int* h_size, int batch_size,
                               int n_sub) {
  divide_by_min_build_index_helper(handle, d_matrix, d_batch, d_index, h_size,
                                   batch_size, n_sub);
}

template <typename DataT>
inline void divide_by_min_execute_helper(const raft::handle_t& handle,
                                         const DataT* d_in, const int* d_batch,
                                         const int* d_index, DataT** hd_out,
                                         int batch_size, int n_sub, int n_obs) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  ML::TimeSeries::divide_by_min_execute(d_in, d_batch, d_index, hd_out,
                                        batch_size, n_sub, n_obs, allocator,
                                        stream);
}

void divide_by_min_execute(const raft::handle_t& handle, const float* d_in,
                           const int* d_batch, const int* d_index,
                           float** hd_out, int batch_size, int n_sub,
                           int n_obs) {
  divide_by_min_execute_helper(handle, d_in, d_batch, d_index, hd_out,
                               batch_size, n_sub, n_obs);
}

void divide_by_min_execute(const raft::handle_t& handle, const double* d_in,
                           const int* d_batch, const int* d_index,
                           double** hd_out, int batch_size, int n_sub,
                           int n_obs) {
  divide_by_min_execute_helper(handle, d_in, d_batch, d_index, hd_out,
                               batch_size, n_sub, n_obs);
}

void divide_by_min_execute(const raft::handle_t& handle, const int* d_in,
                           const int* d_batch, const int* d_index, int** hd_out,
                           int batch_size, int n_sub, int n_obs) {
  divide_by_min_execute_helper(handle, d_in, d_batch, d_index, hd_out,
                               batch_size, n_sub, n_obs);
}

void build_division_map(const raft::handle_t& handle, const int* const* hd_id,
                        const int* h_size, int* d_id_to_pos, int* d_id_to_model,
                        int batch_size, int n_sub) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  ML::TimeSeries::build_division_map(hd_id, h_size, d_id_to_pos, d_id_to_model,
                                     batch_size, n_sub, allocator, stream);
}

template <typename DataT>
inline void merge_series_helper(const raft::handle_t& handle,
                                const DataT* const* hd_in,
                                const int* d_id_to_pos, const int* d_id_to_sub,
                                DataT* d_out, int batch_size, int n_sub,
                                int n_obs) {
  cudaStream_t stream = handle.get_stream();
  auto allocator = handle.get_device_allocator();
  ML::TimeSeries::merge_series(hd_in, d_id_to_pos, d_id_to_sub, d_out,
                               batch_size, n_sub, n_obs, allocator, stream);
}

void merge_series(const raft::handle_t& handle, const float* const* hd_in,
                  const int* d_id_to_pos, const int* d_id_to_sub, float* d_out,
                  int batch_size, int n_sub, int n_obs) {
  merge_series_helper(handle, hd_in, d_id_to_pos, d_id_to_sub, d_out,
                      batch_size, n_sub, n_obs);
}

void merge_series(const raft::handle_t& handle, const double* const* hd_in,
                  const int* d_id_to_pos, const int* d_id_to_sub, double* d_out,
                  int batch_size, int n_sub, int n_obs) {
  merge_series_helper(handle, hd_in, d_id_to_pos, d_id_to_sub, d_out,
                      batch_size, n_sub, n_obs);
}

}  // namespace ML
