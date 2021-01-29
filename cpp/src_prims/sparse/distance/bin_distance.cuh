/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <limits.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <sparse/distance/common.h>
#include <sparse/utils.h>
#include <sparse/distance/ip_distance.cuh>

#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {

// @TODO: Move this into sparse prims (coo_norm)
template <typename value_idx, typename value_t>
__global__ void compute_binary_row_norm_kernel(
  value_t *out, const value_idx *__restrict__ coo_rows,
  const value_t *__restrict__ data, value_idx nnz) {
  value_idx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) {
    // We do conditional here only because it's
    // possible there could be some stray zeros in
    // the sparse structure and removing them would be
    // more expensive.
    atomicAdd(&out[coo_rows[i]], data[i] == 1.0);
  }
}

template <typename value_idx, typename value_t, typename expansion_f>
__global__ void compute_binary_warp_kernel(value_t *__restrict__ C,
                                           const value_t *__restrict__ Q_norms,
                                           const value_t *__restrict__ R_norms,
                                           value_idx n_rows, value_idx n_cols,
                                           expansion_f expansion_func) {
  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx i = tid / n_cols;
  value_idx j = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  value_t q_norm = Q_norms[i];
  value_t r_norm = R_norms[j];
  value_t dot = C[(size_t)i * n_cols + j];
  C[(size_t)i * n_cols + j] = expansion_func(dot, q_norm, r_norm);
}

template <typename value_idx, typename value_t, typename expansion_f,
          int tpb = 1024>
void compute_binary(value_t *C, const value_t *Q_norms, const value_t *R_norms,
                    value_idx n_rows, value_idx n_cols,
                    expansion_f expansion_func, cudaStream_t stream) {
  int blocks = raft::ceildiv<size_t>((size_t)n_rows * n_cols, tpb);
  compute_binary_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, Q_norms, R_norms, n_rows, n_cols, expansion_func);
}

template <typename value_idx, typename value_t, typename expansion_f,
          int tpb = 1024>
void compute_bin_distance(value_t *out, const value_idx *Q_coo_rows,
                          const value_t *Q_data, value_idx Q_nnz,
                          const value_idx *R_coo_rows, const value_t *R_data,
                          value_idx R_nnz, value_idx m, value_idx n,
                          cusparseHandle_t handle,
                          std::shared_ptr<raft::mr::device::allocator> alloc,
                          cudaStream_t stream, expansion_f expansion_func) {
  raft::mr::device::buffer<value_t> Q_norms(alloc, stream, m);
  raft::mr::device::buffer<value_t> R_norms(alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(Q_norms.data(), 0, Q_norms.size() * sizeof(value_t)));
  CUDA_CHECK(
    cudaMemsetAsync(R_norms.data(), 0, R_norms.size() * sizeof(value_t)));

  compute_binary_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_binary_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_binary(out, Q_norms.data(), R_norms.data(), m, n, expansion_func,
                 stream);
}

/**
 * Jaccard distance using the expanded form:
 * 1 - (sum(x_k * y_k) / ((sum(x_k) + sum(y_k)) - sum(x_k * y_k))
 */
template <typename value_idx = int, typename value_t = float>
class jaccard_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit jaccard_expanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing inner products");
    ip_dists.compute(out_dists);

    value_idx *b_indices = ip_dists.b_rows_coo();
    value_t *b_data = ip_dists.b_data_coo();

    CUML_LOG_DEBUG("Computing COO row index array");
    raft::mr::device::buffer<value_idx> search_coo_rows(
      config_->allocator, config_->stream, config_->a_nnz);
    raft::sparse::convert::csr_to_coo(config_->a_indptr, config_->a_nrows,
                                      search_coo_rows.data(), config_->a_nnz,
                                      config_->stream);

    CUML_LOG_DEBUG("Computing Jaccard");
    compute_bin_distance(
      out_dists, search_coo_rows.data(), config_->a_data, config_->a_nnz,
      b_indices, b_data, config_->b_nnz, config_->a_nrows, config_->b_nrows,
      config_->handle, config_->allocator, config_->stream,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
        value_t q_r_union = q_norm + r_norm;
        return 1 - (dot / (q_r_union - dot));
      });
  }

  ~jaccard_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t> *config_;
  raft::mr::device::buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Dice distance using the expanded form:
 * 1 - ((2 * sum(x_k * y_k)) / (sum(x_k)^2 + sum(y_k)^2))
 */
template <typename value_idx = int, typename value_t = float>
class dice_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit dice_expanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing inner products");
    ip_dists.compute(out_dists);

    value_idx *b_indices = ip_dists.b_rows_coo();
    value_t *b_data = ip_dists.b_data_coo();

    CUML_LOG_DEBUG("Computing COO row index array");
    raft::mr::device::buffer<value_idx> search_coo_rows(
      config_->allocator, config_->stream, config_->a_nnz);
    raft::sparse::convert::csr_to_coo(config_->a_indptr, config_->a_nrows,
                                      search_coo_rows.data(), config_->a_nnz,
                                      config_->stream);

    CUML_LOG_DEBUG("Computing Dice dissimilarity");
    compute_bin_distance(
      out_dists, search_coo_rows.data(), config_->a_data, config_->a_nnz,
      b_indices, b_data, config_->b_nnz, config_->a_nrows, config_->b_nrows,
      config_->handle, config_->allocator, config_->stream,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
        value_t q_r_union = (q_norm * q_norm) + (r_norm * r_norm);
        return (2 * dot) / q_r_union;
      });
  }

  ~dice_expanded_distances_t() = default;

 private:
  const distances_config_t<value_idx, value_t> *config_;
  raft::mr::device::buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
