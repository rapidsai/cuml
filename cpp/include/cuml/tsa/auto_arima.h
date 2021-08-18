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

#pragma once

namespace raft {
class handle_t;
}

namespace ML {

/**
 * Batch division by mask step 1: build an index of the position of each series
 * in its new batch and measure the size of each sub-batch
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_mask     Boolean mask
 * @param[out] d_index    Index of each series in its new batch
 * @param[in]  batch_size Batch size
 * @return The number of 'true' series in the mask
 */
int divide_by_mask_build_index(const raft::handle_t& handle,
                               const bool* d_mask,
                               int* d_index,
                               int batch_size);

/**
 * Batch division by mask step 2: create both sub-batches from the mask and
 * index
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_mask     Boolean mask
 * @param[in]  d_index    Index of each series in its new batch
 * @param[out] d_out0     The sub-batch for the 'false' members
 * @param[out] d_out1     The sub-batch for the 'true' members
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 */
void divide_by_mask_execute(const raft::handle_t& handle,
                            const float* d_in,
                            const bool* d_mask,
                            const int* d_index,
                            float* d_out0,
                            float* d_out1,
                            int batch_size,
                            int n_obs);
void divide_by_mask_execute(const raft::handle_t& handle,
                            const double* d_in,
                            const bool* d_mask,
                            const int* d_index,
                            double* d_out0,
                            double* d_out1,
                            int batch_size,
                            int n_obs);
void divide_by_mask_execute(const raft::handle_t& handle,
                            const int* d_in,
                            const bool* d_mask,
                            const int* d_index,
                            int* d_out0,
                            int* d_out1,
                            int batch_size,
                            int n_obs);

/**
 * Batch division by minimum value step 1: build an index of which sub-batch
 * each series belongs to, an index of the position of each series in its new
 * batch, and measure the size of each sub-batch
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_matrix   Matrix of the values to minimize
 *                        Shape: (batch_size, n_sub)
 * @param[out] d_batch    Which sub-batch each series belongs to
 * @param[out] d_index    Index of each series in its new batch
 * @param[out] h_size     Size of each sub-batch (host)
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 */
void divide_by_min_build_index(const raft::handle_t& handle,
                               const float* d_matrix,
                               int* d_batch,
                               int* d_index,
                               int* h_size,
                               int batch_size,
                               int n_sub);
void divide_by_min_build_index(const raft::handle_t& handle,
                               const double* d_matrix,
                               int* d_batch,
                               int* d_index,
                               int* h_size,
                               int batch_size,
                               int n_sub);

/**
 * Batch division by minimum value step 2: create all the sub-batches
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_batch    Which sub-batch each series belongs to
 * @param[in]  d_index    Index of each series in its new sub-batch
 * @param[out] hd_out     Host array of pointers to device arrays of each
 *                        sub-batch
 * @param[in]  batch_size Batch size
 * @param[in]  n_sub      Number of sub-batches
 * @param[in]  n_obs      Number of data points per series
 */
void divide_by_min_execute(const raft::handle_t& handle,
                           const float* d_in,
                           const int* d_batch,
                           const int* d_index,
                           float** hd_out,
                           int batch_size,
                           int n_sub,
                           int n_obs);
void divide_by_min_execute(const raft::handle_t& handle,
                           const double* d_in,
                           const int* d_batch,
                           const int* d_index,
                           double** hd_out,
                           int batch_size,
                           int n_sub,
                           int n_obs);
void divide_by_min_execute(const raft::handle_t& handle,
                           const int* d_in,
                           const int* d_batch,
                           const int* d_index,
                           int** hd_out,
                           int batch_size,
                           int n_sub,
                           int n_obs);

/**
 * Build a map to associate each batch member with a model and index in the
 * associated sub-batch
 *
 * @param[in]  handle        cuML handle
 * @param[in]  hd_id         Host array of pointers to device arrays containing
 *                           the indices of the members of each sub-batch
 * @param[in]  h_size        Host array containing the size of each sub-batch
 * @param[out] d_id_to_pos   Device array containing the position of each
 *                           member in its new sub-batch
 * @param[out] d_id_to_model Device array associating each member with its
 *                           sub-batch
 * @param[in]  batch_size    Batch size
 * @param[in]  n_sub         Number of sub-batches
 */
void build_division_map(const raft::handle_t& handle,
                        const int* const* hd_id,
                        const int* h_size,
                        int* d_id_to_pos,
                        int* d_id_to_model,
                        int batch_size,
                        int n_sub);

/**
 * Merge multiple sub-batches into one batch according to the maps that
 * associate each id in the unique batch to a sub-batch and a position in
 * this sub-batch.
 *
 * @param[in]  handle        cuML handle
 * @param[in]  hd_in       Host array of pointers to device arrays containing
 *                         the sub-batches
 * @param[in]  d_id_to_pos Device array containing the position of each member
 *                         in its new sub-batch
 * @param[in]  d_id_to_sub Device array associating each member with its
 *                         sub-batch
 * @param[out] d_out       Output merged batch
 * @param[in]  batch_size  Batch size
 * @param[in]  n_sub       Number of sub-batches
 * @param[in]  n_obs       Number of observations (or forecasts) per series
 */
void merge_series(const raft::handle_t& handle,
                  const float* const* hd_in,
                  const int* d_id_to_pos,
                  const int* d_id_to_sub,
                  float* d_out,
                  int batch_size,
                  int n_sub,
                  int n_obs);
void merge_series(const raft::handle_t& handle,
                  const double* const* hd_in,
                  const int* d_id_to_pos,
                  const int* d_id_to_sub,
                  double* d_out,
                  int batch_size,
                  int n_sub,
                  int n_obs);

}  // namespace ML
