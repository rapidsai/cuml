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

#include <common/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>
#include <sparse/csr.cuh>

#include <cusparse_v2.h>
#include <raft/sparse/cusparse_wrappers.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

template <typename value_idx, typename value_t>
struct distances_config_t {
  // left side
  value_idx index_nrows;
  value_idx index_ncols;
  value_idx index_nnz;
  value_idx *csc_index_indptr;
  value_idx *csc_index_indices;
  value_t *csc_index_data;

  // right side
  value_idx search_nrows;
  value_idx search_ncols;
  value_idx search_nnz;
  value_idx *csr_search_indptr;
  value_idx *csr_search_indices;
  value_t *csr_search_data;

  cusparseHandle_t handle;

  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;
};

/**
 * Simple inner product distance with sparse matrix multiply
 */
template <typename value_idx = int, typename value_t = float>
struct ip_distances_t {
  explicit ip_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      alpha(1.0) {
    init_mat_descriptor(matA);
    init_mat_descriptor(matB);
    init_mat_descriptor(matC);
    init_mat_descriptor(matD);

    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));

    CUSPARSE_CHECK(
      cusparseSetPointerMode(config.handle, CUSPARSE_POINTER_MODE_HOST));
  }

  void compute(value_t *out_distances) {

	  /**
	   * Compute pairwise distances
	   */
      device_buffer<value_idx> out_batch_indptr(config_.allocator, config_.stream,
                                                config_.search_nrows + 1);
      device_buffer<value_idx> out_batch_indices(config_.allocator, config_.stream, 0);
      device_buffer<value_t> out_batch_data(config_.allocator, config_.stream, 0);

      value_idx out_batch_nnz = get_nnz(out_batch_indptr.data());

      out_batch_indices.resize(out_batch_nnz, config_.stream);
      out_batch_data.resize(out_batch_nnz, config_.stream);

      compute(out_batch_indptr.data(), out_batch_indices.data(),
                            out_batch_data.data());

      /**
       * Convert output to dense
       */
      csr_to_dense(config_.handle, config_.search_nrows,
                   config_.index_nrows, out_batch_indptr.data(),
                   out_batch_indices.data(), out_batch_data.data(), true,
				   out_distances, config_.stream);
  }

  ~ip_distances_t() {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matA));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matB));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matC));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matD));
  }

 private:

  void init_mat_descriptor(cusparseMatDescr_t &mat) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL));
  }

  value_idx get_nnz(value_idx *csr_out_indptr) {
    value_idx m = config_.search_nrows, n = config_.index_nrows,
              k = config_.search_ncols;

    size_t workspace_size;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2_buffersizeext<value_t>(
      config_.handle, m, n, k, &alpha, NULL, matA, config_.search_nnz,
      config_.csr_search_indptr, config_.csr_search_indices, matB,
      config_.index_nnz, config_.csc_index_indptr, config_.csc_index_indices,
      matD, 0, NULL, NULL, info, &workspace_size, config_.stream));

    workspace.resize(workspace_size, config_.stream);

    value_idx out_nnz = 0;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2nnz(
      config_.handle, m, n, k, matA, config_.search_nnz,
      config_.csr_search_indptr, config_.csr_search_indices, matB,
      config_.index_nnz, config_.csc_index_indptr, config_.csc_index_indices,
      matD, 0, NULL, NULL, matC, csr_out_indptr, &out_nnz, info,
      workspace.data(), config_.stream));

    return out_nnz;
  }

  void compute(const value_idx *csr_out_indptr, value_idx *csr_out_indices,
               value_t *csr_out_data) {
    value_idx m = config_.search_nrows, n = config_.index_nrows,
              k = config_.search_ncols;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2<value_t>(
      config_.handle, m, n, k, &alpha, matA, config_.search_nnz,
      config_.csr_search_data, config_.csr_search_indptr,
      config_.csr_search_indices, matB, config_.index_nnz,
      config_.csc_index_data, config_.csc_index_indptr,
      config_.csc_index_indices, NULL, matD, 0, NULL, NULL, NULL, matC,
      csr_out_data, csr_out_indptr, csr_out_indices, info, workspace.data(),
      config_.stream));
  }

  value_t alpha;
  csrgemm2Info_t info;
  cusparseMatDescr_t matA;
  cusparseMatDescr_t matB;
  cusparseMatDescr_t matC;
  cusparseMatDescr_t matD;
  device_buffer<char> workspace;
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx, typename value_t>
__global__ void compute_sq_norm_kernel(value_t *out, value_idx *csr_indices,
                                       value_t *csr_data, value_idx nnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < nnz) atomicAdd(&out[csr_indices[i]], csr_data[i] * csr_data[i]);
}

template <typename value_idx, typename value_t>
__global__ void compute_euclidean_kernel(value_t *C, value_t *Q_sq_norms,
                                         value_t *R_sq_norms, value_idx n_rows,
                                         value_idx n_cols) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int i = index / n_cols;
  int j = index % n_cols;

  // Cuda store row major.
  if (i < n_rows && j < n_cols)
    C[i * n_cols + j] = Q_sq_norms[i] - 2.0 * C[i * n_cols + j] + R_sq_norms[j];
}

template <typename value_idx, typename value_t, int tpb = 256>
void compute_l2(value_t *out, value_idx *Q_csr_indices, value_t *Q_csr_data,
                value_idx Q_nnz, value_idx *R_csr_indices, value_t *R_csr_data,
                value_idx R_nnz, value_idx m, value_idx n,
                cusparseHandle_t handle, std::shared_ptr<deviceAllocator> alloc,
                cudaStream_t stream) {
  device_buffer<value_t> Q_sq_norms(alloc, stream, m);
  device_buffer<value_t> R_sq_norms(alloc, stream, n);

  get_col_norms<<<ceildiv(Q_nnz, tpb), tpb, 0, stream>>>(
    Q_sq_norms.data(), Q_csr_indices, Q_csr_data, Q_nnz);
  get_col_norms<<<ceildiv(R_nnz, tpb), tpb, 0, stream>>>(
    R_sq_norms.data(), R_csr_indices, R_csr_data, R_nnz);

  add_norms<<<ceildiv(m * n, tpb), tpb, 0, stream>>>(out, Q_sq_norms.data(),
                                                     R_sq_norms.data(), m, n);
}

/**
 * L2 distance using the expanded form: sum(x_k)^2 + sum(y_k)^2 - 2 * sum(x_k * y_k)
 */
template <typename value_idx = int, typename value_t = float>
struct l2_distances_t {
  explicit l2_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config),
      workspace(config.allocator, config.stream, 0),
      ip_dists(config) {}

    //TODO: Modify interface to just output dense directly and just make a compute call.
    void compute(value_t *out_dists) {


    }


 private:
  distances_config_t<value_idx, value_t> config_;
  device_buffer<char> workspace;
  ip_distances_t<value_idx, value_t> ip_dists;
};




};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
