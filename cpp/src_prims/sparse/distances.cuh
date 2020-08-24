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
    : config_(config), workspace(config.allocator, config.stream, 0) {
    alpha = 1.0;
    beta = 0.0;

    CUSPARSE_CHECK(cusparseCreateMatDescr(&matA));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&matB));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&matC));

    CUSPARSE_CHECK(cusparseSetMatIndexBase(matA, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(matB, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(matC, CUSPARSE_INDEX_BASE_ZERO));

    CUSPARSE_CHECK(cusparseSetMatType(matA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatType(matB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatType(matC, CUSPARSE_MATRIX_TYPE_GENERAL));

    CUSPARSE_CHECK(cusparseCreateCsrgemm2Info(&info));

    CUSPARSE_CHECK(
      cusparseSetPointerMode(config.handle, CUSPARSE_POINTER_MODE_HOST));
  }

  value_idx get_nnz(value_idx *csr_out_indptr) {
    value_idx m = config_.search_nrows, n = config_.search_ncols,
              k = config_.index_nrows;

    size_t workspace_size;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2_buffersizeext(
      config_.handle, m, n, k, &alpha, &beta, matA, config_.search_nnz,
      config_.csr_search_indptr, config_.csr_search_indices, matB,
      config_.index_nnz, config_.csc_index_indptr, config_.csc_index_indices,
      NULL, 0, NULL, NULL, info, &workspace_size, config_.stream));

    workspace.resize(workspace_size, config_.stream);

    value_idx out_nnz;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2nnz(
      config_.handle, m, n, k, matA, config_.search_nnz,
      config_.csr_search_indptr, config_.csr_search_indices, matB,
      config_.index_nnz, config_.csc_index_indptr, config_.csc_index_indices,
      NULL, 0, NULL, NULL, matC, csr_out_indptr, &out_nnz, info,
      workspace.data(), config_.stream));

    return out_nnz;
  }

  void compute(value_idx *csr_out_indptr, value_idx *csr_out_indices,
               value_t *csr_out_data) {
    value_idx m = config_.search_nrows, n = config_.search_ncols,
              k = config_.index_rows;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2(
      config_.handle, m, n, k, &alpha, matA, config_.search_nnz,
      config_.csr_search_data, config_.csr_search_rowind,
      config_.csr_search_indices, matB, config_.index_nna,
      config_.csc_index_data, config_.csc_index_rowind,
      config_.csc_index_indices, &beta, NULL, 0, NULL, NULL, NULL, matC,
      config_.csr_out_data, config_.csr_out_rowind, config_.csr_out_indices,
      info, workspace.data()));
  }

  ~ip_distances_t() {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matA));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matB));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matC));
  }

 private:
  value_t alpha;
  value_t beta;
  csrgemm2Info_t info;
  cusparseMatDescr_t matA;
  cusparseMatDescr_t matB;
  cusparseMatDescr_t matC;
  device_buffer<char> workspace;
  distances_config_t<value_idx, value_t> config_;
};

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
