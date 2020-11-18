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

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <common/device_buffer.hpp>

#include <sparse/utils.h>
#include <sparse/csr.cuh>

#include <cuml/common/cuml_allocator.hpp>
#include <cuml/neighbors/knn.hpp>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

template <typename value_idx, typename value_t>
struct distances_config_t {
  // left side
  value_idx a_nrows;
  value_idx a_ncols;
  value_idx a_nnz;
  value_idx *a_indptr;
  value_idx *a_indices;
  value_t *a_data;

  // right side
  value_idx b_nrows;
  value_idx b_ncols;
  value_idx b_nnz;
  value_idx *b_indptr;
  value_idx *b_indices;
  value_t *b_data;

  cusparseHandle_t handle;

  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
};

template <typename value_t>
class distances_t {
 public:
  virtual void compute(value_t *out) { CUML_LOG_DEBUG("INside base"); }
  virtual ~distances_t() = default;
};

/**
 * Simple inner product distance with sparse matrix multiply
 */
template <typename value_idx = int, typename value_t = float>
class ip_distances_t : public distances_t<value_t> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   */
  explicit ip_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      csc_indptr(config.allocator, config.stream, 0),
      csc_indices(config.allocator, config.stream, 0),
      csc_data(config.allocator, config.stream, 0),
      alpha(1.0) {
    init_mat_descriptor(matA);
    init_mat_descriptor(matB);
    init_mat_descriptor(matC);
    init_mat_descriptor(matD);

    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));

    CUSPARSE_CHECK(cusparseGetPointerMode(config.handle, &orig_ptr_mode));

    CUSPARSE_CHECK(
      cusparseSetPointerMode(config.handle, CUSPARSE_POINTER_MODE_HOST));
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(value_t *out_distances) {
    /**
	   * Compute pairwise distances and return dense matrix in column-major format
	   */

    CUML_LOG_DEBUG("Compute() inside inner-product d");
    device_buffer<value_idx> out_batch_indptr(config_.allocator, config_.stream,
                                              config_.a_nrows + 1);
    device_buffer<value_idx> out_batch_indices(config_.allocator,
                                               config_.stream, 0);
    device_buffer<value_t> out_batch_data(config_.allocator, config_.stream, 0);

    value_idx out_batch_nnz = get_nnz(out_batch_indptr.data());

    out_batch_indices.resize(out_batch_nnz, config_.stream);
    out_batch_data.resize(out_batch_nnz, config_.stream);

    compute_gemm(out_batch_indptr.data(), out_batch_indices.data(),
                 out_batch_data.data());

    /**
     * Convert output to dense
     * TODO: This is wasteful of memory and adds additional latency.
     * It would be nice if there was a gemm that could do
     * (sparse, sparse)->dense natively.
     */
    csr_to_dense(config_.handle, config_.a_nrows, config_.b_nrows,
                 out_batch_indptr.data(), out_batch_indices.data(),
                 out_batch_data.data(), config_.a_nrows, out_distances,
                 config_.stream, true);
  }

  value_idx *trans_indptr() { return csc_indptr.data(); }

  value_idx *trans_indices() { return csc_indices.data(); }

  value_t *trans_data() { return csc_data.data(); }

  ~ip_distances_t() {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matA));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matB));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matC));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matD));

    CUSPARSE_CHECK_NO_THROW(
      cusparseSetPointerMode(config_.handle, orig_ptr_mode));
  }

 private:
  void init_mat_descriptor(cusparseMatDescr_t &mat) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL));
  }

  value_idx get_nnz(value_idx *csr_out_indptr) {
    value_idx m = config_.a_nrows, n = config_.b_nrows, k = config_.a_ncols;

    transpose_b();

    size_t workspace_size;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2_buffersizeext<value_t>(
      config_.handle, m, n, k, &alpha, NULL, matA, config_.a_nnz,
      config_.a_indptr, config_.a_indices, matB, config_.b_nnz,
      csc_indptr.data(), csc_indices.data(), matD, 0, NULL, NULL, info,
      &workspace_size, config_.stream));

    workspace.resize(workspace_size, config_.stream);

    value_idx out_nnz = 0;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2nnz(
      config_.handle, m, n, k, matA, config_.a_nnz, config_.a_indptr,
      config_.a_indices, matB, config_.b_nnz, csc_indptr.data(),
      csc_indices.data(), matD, 0, NULL, NULL, matC, csr_out_indptr, &out_nnz,
      info, workspace.data(), config_.stream));

    return out_nnz;
  }

  void compute_gemm(const value_idx *csr_out_indptr, value_idx *csr_out_indices,
                    value_t *csr_out_data) {
    value_idx m = config_.a_nrows, n = config_.b_nrows, k = config_.a_ncols;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2<value_t>(
      config_.handle, m, n, k, &alpha, matA, config_.a_nnz, config_.a_data,
      config_.a_indptr, config_.a_indices, matB, config_.b_nnz, csc_data.data(),
      csc_indptr.data(), csc_indices.data(), NULL, matD, 0, NULL, NULL, NULL,
      matC, csr_out_data, csr_out_indptr, csr_out_indices, info,
      workspace.data(), config_.stream));
  }

  void transpose_b() {
    /**
     * Transpose index array into csc
     */
    CUML_LOG_DEBUG("Transposing index CSR. rows=%d, cols=%d, nnz=%d",
                   config_.b_nrows, config_.b_ncols, config_.b_nnz);

    csc_indptr.resize(config_.b_ncols + 1, config_.stream);
    csc_indices.resize(config_.b_nnz, config_.stream);
    csc_data.resize(config_.b_nnz, config_.stream);

    csr_transpose(config_.handle, config_.b_indptr, config_.b_indices,
                  config_.b_data, csc_indptr.data(), csc_indices.data(),
                  csc_data.data(), config_.b_nrows, config_.b_ncols,
                  config_.b_nnz, config_.allocator, config_.stream);
  }

  value_t alpha;
  csrgemm2Info_t info;
  cusparseMatDescr_t matA;
  cusparseMatDescr_t matB;
  cusparseMatDescr_t matC;
  cusparseMatDescr_t matD;
  cusparsePointerMode_t orig_ptr_mode;
  device_buffer<char> workspace;
  device_buffer<value_idx> csc_indptr;
  device_buffer<value_idx> csc_indices;
  device_buffer<value_t> csc_data;
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx, typename value_t>
__global__ void compute_sq_norm_kernel(value_t *out, const value_idx *coo_rows,
                                       const value_t *data, value_idx nnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nnz) {
    atomicAdd(&out[coo_rows[i]], data[i] * data[i]);
  }
}

template <typename value_idx, typename value_t>
__global__ void compute_euclidean_warp_kernel(value_t *C,
                                              const value_t *Q_sq_norms,
                                              const value_t *R_sq_norms,
                                              value_idx n_cols) {
  value_idx i = blockIdx.x;
  value_idx tid = threadIdx.x;

  __shared__ value_t q_norm;

  if (tid == 0) {
    q_norm = Q_sq_norms[i];
  }

  __syncthreads();

  for (int j = tid; j < n_cols; j += blockDim.x) {
    value_t r_norm = R_sq_norms[j];
    value_t dot = C[i * n_cols + j];

    value_t val = q_norm + r_norm - 2.0 * dot;
    if (fabsf(val) < 0.0001) val = 0.0;

    C[i * n_cols + j] = val;
  }
}

template <typename value_idx, typename value_t>
void compute_euclidean(value_t *C, const value_t *Q_sq_norms,
                       const value_t *R_sq_norms, value_idx n_rows,
                       value_idx n_cols, cudaStream_t stream) {
  int blockdim = block_dim(n_cols);

  compute_euclidean_warp_kernel<<<n_rows, blockdim, 0, stream>>>(
    C, Q_sq_norms, R_sq_norms, n_cols);
}

template <typename value_idx, typename value_t, int tpb = 256>
void compute_l2(value_t *out, const value_idx *Q_coo_rows,
                const value_t *Q_data, value_idx Q_nnz,
                const value_idx *R_coo_rows, const value_t *R_data,
                value_idx R_nnz, value_idx m, value_idx n,
                cusparseHandle_t handle, std::shared_ptr<deviceAllocator> alloc,
                cudaStream_t stream) {
  device_buffer<value_t> Q_sq_norms(alloc, stream, m);
  device_buffer<value_t> R_sq_norms(alloc, stream, n);
  CUDA_CHECK(
    cudaMemsetAsync(Q_sq_norms.data(), 0, Q_sq_norms.size() * sizeof(value_t)));
  CUDA_CHECK(
    cudaMemsetAsync(R_sq_norms.data(), 0, R_sq_norms.size() * sizeof(value_t)));

  compute_sq_norm_kernel<<<raft::ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_coo_rows, Q_data, Q_nnz);
  compute_sq_norm_kernel<<<raft::ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_coo_rows, R_data, R_nnz);

  compute_euclidean(out, Q_sq_norms.data(), R_sq_norms.data(), m, n, stream);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 * The expanded form is more efficient for sparse data.
 */
template <typename value_idx = int, typename value_t = float>
class l2_distances_t : public distances_t<value_t> {
 public:
  explicit l2_distances_t(distances_config_t<value_idx, value_t> config)
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
    compute_l2(out_dists, search_coo_rows.data(), config_.a_data, config_.a_nnz,
               b_indices, b_data, config_.b_nnz, config_.a_nrows,
               config_.b_nrows, config_.handle, config_.allocator,
               config_.stream);
    CUML_LOG_DEBUG("Done.");
  }

  ~l2_distances_t() = default;

 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};

/**
 * Compute pairwise distances between A and B, using the provided
 * input configuration and distance function.
 *
 * @tparam value_idx index type
 * @tparam value_t value type
 * @param[out] out dense output array (size A.nrows * B.nrows)
 * @param[in] input_config input argument configuration
 * @param[in] metric distance metric to use
 */
template class ip_distances_t<int, float>;
template class l2_distances_t<int, float>;
template class distances_config_t<int, float>;

template <typename value_idx = int, typename value_t = float>
void pairwiseDistance(value_t *out,
                      distances_config_t<value_idx, value_t> input_config,
                      raft::distance::DistanceType metric) {
  CUML_LOG_DEBUG("Running sparse pairwise distances with metric=%d", metric);

  switch (metric) {
    case raft::distance::DistanceType::EucExpandedL2:
      // EucExpandedL2
      l2_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    case raft::distance::DistanceType::InnerProduct:
      // InnerProduct
      ip_distances_t<value_idx, value_t>(input_config).compute(out);
      break;
    default:
      THROW("Unsupported metric: %d", metric);
  }
}

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
