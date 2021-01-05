/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <sparse/distance/common.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>

#include <common/device_buffer.hpp>

#include <sparse/utils.h>
#include <sparse/csr.cuh>

#include <sparse/distance/common.h>
#include <sparse/distance/ip_distance.cuh>

#include <cuml/common/cuml_allocator.hpp>
#include <cuml/neighbors/knn.hpp>

#include <nvfunctional>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

// @TODO: Move this into sparse prims (coo_norm)
template <typename value_idx, typename value_t>
__global__ void compute_row_norm_kernel(value_t *out, const value_idx *coo_rows,
                                        const value_t *data, value_idx nnz,
                                        float norm = 2.0) {
  value_idx i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nnz) {
    atomicAdd(&out[coo_rows[i]], powf(data[i], norm));
  }
}

template <typename value_idx, typename value_t, typename expansion_f>
__global__ void compute_euclidean_warp_kernel(
  value_t *C, const value_t *Q_sq_norms, const value_t *R_sq_norms,
  value_idx n_rows, value_idx n_cols, expansion_f expansion_func) {
  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx i = tid / n_cols;
  value_idx j = tid % n_cols;

  if (i >= n_rows || j >= n_cols) return;

  value_t q_norm = Q_sq_norms[i];
  value_t r_norm = R_sq_norms[j];
  value_t dot = C[i * n_cols + j];

  // e.g. Euclidean expansion func = -2.0 * dot + q_norm + r_norm
  value_t val = expansion_func(dot, q_norm, r_norm);

  // correct for small instabilities
  if (fabsf(val) < 0.0001) val = 0.0;

  C[i * n_cols + j] = val;
}

template <typename value_idx, typename value_t, int tpb = 1024,
          typename expansion_f>
void compute_euclidean(value_t *C, const value_t *Q_sq_norms,
                       const value_t *R_sq_norms, value_idx n_rows,
                       value_idx n_cols, cudaStream_t stream,
                       expansion_f expansion_func) {
  int blocks = raft::ceildiv(n_rows * n_cols, tpb);
  compute_euclidean_warp_kernel<<<blocks, tpb, 0, stream>>>(
    C, Q_sq_norms, R_sq_norms, n_rows, n_cols, expansion_func);
}

template <typename value_idx, typename value_t, int tpb = 1024,
          typename expansion_f>
void compute_l2(value_t *out, const value_idx *Q_coo_rows,
                const value_t *Q_data, value_idx Q_nnz,
                const value_idx *R_coo_rows, const value_t *R_data,
                value_idx R_nnz, value_idx m, value_idx n,
                cusparseHandle_t handle, std::shared_ptr<deviceAllocator> alloc,
                cudaStream_t stream, expansion_f expansion_func) {
  device_buffer<value_t> Q_sq_norms(alloc, stream, m);
  device_buffer<value_t> R_sq_norms(alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(Q_sq_norms.data(), 0, Q_sq_norms.size() * sizeof(value_t)));
  CUDA_CHECK(
    cudaMemsetAsync(R_sq_norms.data(), 0, R_sq_norms.size() * sizeof(value_t)));

  compute_row_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_row_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_euclidean(out, Q_sq_norms.data(), R_sq_norms.data(), m, n, stream,
                    expansion_func);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class l2_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit l2_expanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing inner products");
    ip_dists.compute(out_dists);

    value_idx *b_indices = ip_dists.trans_indices();
    value_t *b_data = ip_dists.trans_data();

    CUML_LOG_DEBUG("Computing COO row index array");
    device_buffer<value_idx> search_coo_rows(config_.allocator, config_.stream,
                                             config_.a_nnz);
    csr_to_coo(config_.a_indptr, config_.a_nrows, search_coo_rows.data(),
               config_.a_nnz, config_.stream);

    CUML_LOG_DEBUG("Done.");

    CUML_LOG_DEBUG("Computing L2");
    compute_l2(
      out_dists, search_coo_rows.data(), config_.a_data, config_.a_nnz,
      b_indices, b_data, config_.b_nnz, config_.a_nrows, config_.b_nrows,
      config_.handle, config_.allocator, config_.stream,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
        return -2 * dot + q_norm + r_norm;
      });
    CUML_LOG_DEBUG("Done.");
  }

  ~l2_expanded_distances_t() = default;

 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Cosine distance using the expanded form: 1 - ( sum(x_k * y_k) / (sqrt(sum(x_k)^2) * sqrt(sum(y_k)^2)))
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class cosine_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit cosine_expanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing inner products");
    ip_dists.compute(out_dists);

    value_idx *b_indices = ip_dists.trans_indices();
    value_t *b_data = ip_dists.trans_data();

    CUML_LOG_DEBUG("Computing COO row index array");
    device_buffer<value_idx> search_coo_rows(config_.allocator, config_.stream,
                                             config_.a_nnz);
    csr_to_coo(config_.a_indptr, config_.a_nrows, search_coo_rows.data(),
               config_.a_nnz, config_.stream);

    CUML_LOG_DEBUG("Done.");

    CUML_LOG_DEBUG("Computing L2");
    compute_l2(
      out_dists, search_coo_rows.data(), config_.a_data, config_.a_nnz,
      b_indices, b_data, config_.b_nnz, config_.a_nrows, config_.b_nrows,
      config_.handle, config_.allocator, config_.stream,
      [] __device__ __host__(value_t dot, value_t q_norm, value_t r_norm) {
        value_t q_normalized = sqrt(q_norm);
        value_t r_normalized = sqrt(r_norm);
        value_t cos = dot / (q_normalized * r_normalized);
        return 1 - cos;
      });
    CUML_LOG_DEBUG("Done.");
  }

  ~cosine_expanded_distances_t() = default;

 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Cosine distance using the expanded form: 1 - ( sum(x_k * y_k) / (sqrt(sum(x_k)^2) * sqrt(sum(y_k)^2)))
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class hellinger_expanded_distances_t : public distances_t<value_t> {
 public:
  explicit hellinger_expanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      l2_dists(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Computing Hellinger Distance");

    // First sqrt A and B
    raft::linalg::unaryOp<value_t>(
      config_.a_data, config_.a_data, config_.a_nnz,
      [=] __device__(value_t input) { return sqrt(input); }, config_.stream);
    raft::linalg::unaryOp<value_t>(
      config_.b_data, config_.b_data, config_.b_nnz,
      [=] __device__(value_t input) { return sqrt(input); }, config_.stream);

    l2_dists.compute(out_dists);

    // Revert sqrt of A and B
    raft::linalg::unaryOp<value_t>(
      config_.a_data, config_.a_data, config_.a_nnz,
      [=] __device__(value_t input) { return powf(input, 2.0); },
      config_.stream);
    raft::linalg::unaryOp<value_t>(
      config_.b_data, config_.b_data, config_.b_nnz,
      [=] __device__(value_t input) { return powf(input, 2.0); },
      config_.stream);

    CUML_LOG_DEBUG("Done.");
  }

  ~hellinger_expanded_distances_t() = default;

 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  l2_expanded_distances_t<value_idx, value_t> l2_dists;
};

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
