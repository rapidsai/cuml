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
#include <sparse/linalg/transpose.h>
#include <sparse/utils.h>
#include <sparse/convert/csr.cuh>
#include <sparse/convert/dense.cuh>
#include <sparse/distance/coo_spmv.cuh>
#include <sparse/distance/operators.cuh>

#include <nvfunctional>

#include <cusparse_v2.h>

namespace raft {
namespace sparse {
namespace distance {

/**
 * A simple interface that enables different instances
 * of inner product. Currently, there are two implementations:
 * cusparse gemm and our own semiring spmv.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
class ip_trans_getters_t : public distances_t<value_t> {
 public:
  /**
   * A copy of B's data in coo format. This is
   * useful for downstream distances that
   * might be able to compute a norm instead of
   * point-wise products.
   * @return
   */
  virtual value_t *b_data_coo() = 0;

  /**
   * A copy of B's rows in coo format. This is
   * useful for downstream distances that
   * might be able to compute a norm instead of
   * point-wise products.
   * @return
   */
  virtual value_idx *b_rows_coo() = 0;

  virtual ~ip_trans_getters_t() = default;
};

/**
 * Simple inner product distance with sparse matrix multiply. This
 * uses cusparse and requires both B to be transposed as well as
 * the output to be explicitly converted to dense form (which requires
 * 3 copies of the dense data- 2 for the cusparse csr output and
 * 1 for the final m*n dense matrix.)
 */
template <typename value_idx, typename value_t>
class ip_distances_gemm_t : public ip_trans_getters_t<value_idx, value_t> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   *
   * TODO: Remove this once we have a semiring SPGEMM
   * Ref: https://github.com/rapidsai/cuml/issues/3371
   */
  explicit ip_distances_gemm_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config),
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
    raft::mr::device::buffer<value_idx> out_batch_indptr(
      config_->allocator, config_->stream, config_->a_nrows + 1);
    raft::mr::device::buffer<value_idx> out_batch_indices(config_->allocator,
                                                          config_->stream, 0);
    raft::mr::device::buffer<value_t> out_batch_data(config_->allocator,
                                                     config_->stream, 0);

    value_idx out_batch_nnz = get_nnz(out_batch_indptr.data());

    out_batch_indices.resize(out_batch_nnz, config_->stream);
    out_batch_data.resize(out_batch_nnz, config_->stream);

    compute_gemm(out_batch_indptr.data(), out_batch_indices.data(),
                 out_batch_data.data());

    raft::sparse::convert::csr_to_dense(
      config_->handle, config_->a_nrows, config_->b_nrows,
      out_batch_indptr.data(), out_batch_indices.data(), out_batch_data.data(),
      config_->a_nrows, out_distances, config_->stream, true);
  }

  virtual value_idx *b_rows_coo() { return csc_indices.data(); }

  value_t *b_data_coo() { return csc_data.data(); }

  ~ip_distances_gemm_t() {
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matA));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matB));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matC));
    CUSPARSE_CHECK_NO_THROW(cusparseDestroyMatDescr(matD));

    CUSPARSE_CHECK_NO_THROW(
      cusparseSetPointerMode(config_->handle, orig_ptr_mode));
  }

 private:
  void init_mat_descriptor(cusparseMatDescr_t &mat) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&mat));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL));
  }

  value_idx get_nnz(value_idx *csr_out_indptr) {
    value_idx m = config_->a_nrows, n = config_->b_nrows, k = config_->a_ncols;

    transpose_b();

    size_t workspace_size;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2_buffersizeext<value_t>(
      config_->handle, m, n, k, &alpha, NULL, matA, config_->a_nnz,
      config_->a_indptr, config_->a_indices, matB, config_->b_nnz,
      csc_indptr.data(), csc_indices.data(), matD, 0, NULL, NULL, info,
      &workspace_size, config_->stream));

    workspace.resize(workspace_size, config_->stream);

    value_idx out_nnz = 0;

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2nnz(
      config_->handle, m, n, k, matA, config_->a_nnz, config_->a_indptr,
      config_->a_indices, matB, config_->b_nnz, csc_indptr.data(),
      csc_indices.data(), matD, 0, NULL, NULL, matC, csr_out_indptr, &out_nnz,
      info, workspace.data(), config_->stream));

    return out_nnz;
  }

  void compute_gemm(const value_idx *csr_out_indptr, value_idx *csr_out_indices,
                    value_t *csr_out_data) {
    value_idx m = config_->a_nrows, n = config_->b_nrows, k = config_->a_ncols;

    int start = raft::curTimeMillis();

    CUDA_CHECK(cudaStreamSynchronize(config_->stream));

    CUSPARSE_CHECK(raft::sparse::cusparsecsrgemm2<value_t>(
      config_->handle, m, n, k, &alpha, matA, config_->a_nnz, config_->a_data,
      config_->a_indptr, config_->a_indices, matB, config_->b_nnz,
      csc_data.data(), csc_indptr.data(), csc_indices.data(), NULL, matD, 0,
      NULL, NULL, NULL, matC, csr_out_data, csr_out_indptr, csr_out_indices,
      info, workspace.data(), config_->stream));

    CUDA_CHECK(cudaStreamSynchronize(config_->stream));

    printf("CSR GEMM TOOK: %dms\n", raft::curTimeMillis() - start);
  }

  void transpose_b() {
    /**
     * Transpose index array into csc
     */
    CUML_LOG_DEBUG("Transposing index CSR. rows=%d, cols=%d, nnz=%d",
                   config_->b_nrows, config_->b_ncols, config_->b_nnz);

    csc_indptr.resize(config_->b_ncols + 1, config_->stream);
    csc_indices.resize(config_->b_nnz, config_->stream);
    csc_data.resize(config_->b_nnz, config_->stream);

    raft::sparse::linalg::csr_transpose(
      config_->handle, config_->b_indptr, config_->b_indices, config_->b_data,
      csc_indptr.data(), csc_indices.data(), csc_data.data(), config_->b_nrows,
      config_->b_ncols, config_->b_nnz, config_->allocator, config_->stream);
  }

  value_t alpha;
  csrgemm2Info_t info;
  cusparseMatDescr_t matA;
  cusparseMatDescr_t matB;
  cusparseMatDescr_t matC;
  cusparseMatDescr_t matD;
  cusparsePointerMode_t orig_ptr_mode;
  raft::mr::device::buffer<char> workspace;
  raft::mr::device::buffer<value_idx> csc_indptr;
  raft::mr::device::buffer<value_idx> csc_indices;
  raft::mr::device::buffer<value_t> csc_data;
  const distances_config_t<value_idx, value_t> *config_;
};

template <typename value_idx, typename value_t>
class ip_distances_spmv_t : public ip_trans_getters_t<value_idx, value_t> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   */
  ip_distances_spmv_t(const distances_config_t<value_idx, value_t> &config)
    : config_(&config),
      coo_rows_b(config.allocator, config.stream, config.b_nnz) {
    raft::sparse::convert::csr_to_coo(config_->b_indptr, config_->b_nrows,
                                      coo_rows_b.data(), config_->b_nnz,
                                      config_->stream);
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(value_t *out_distances) {
    /**
	   * Compute pairwise distances and return dense matrix in row-major format
	   */
    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_distances, *config_, coo_rows_b.data(), Product(), Sum(),
      AtomicAdd());
  }

  value_idx *b_rows_coo() { return coo_rows_b.data(); }

  value_t *b_data_coo() { return config_->b_data; }

  ~ip_distances_spmv_t() = default;

 private:
  const distances_config_t<value_idx, value_t> *config_;
  raft::mr::device::buffer<value_idx> coo_rows_b;
};

template <typename value_idx = int, typename value_t = float>
class ip_distances_t : public distances_t<value_t> {
 public:
  /**
   * Computes simple sparse inner product distances as sum(x_y * y_k)
   * @param[in] config specifies inputs, outputs, and sizes
   */
  explicit ip_distances_t(const distances_config_t<value_idx, value_t> &config)
    : config_(&config) {
    if (config_->a_ncols < max_cols_per_block<value_idx, value_t>()) {
      internal_ip_dist =
        std::make_unique<ip_distances_spmv_t<value_idx, value_t>>(*config_);
    } else {
      internal_ip_dist =
        std::make_unique<ip_distances_gemm_t<value_idx, value_t>>(*config_);
    }
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(value_t *out_distances) {
    /**
	   * Compute pairwise distances and return dense matrix in column-major format
	   */
    internal_ip_dist->compute(out_distances);
  }

  virtual value_idx *b_rows_coo() const {
    return internal_ip_dist->b_rows_coo();
  }

  virtual value_t *b_data_coo() const { return internal_ip_dist->b_data_coo(); }

 private:
  const distances_config_t<value_idx, value_t> *config_;
  std::unique_ptr<ip_trans_getters_t<value_idx, value_t>> internal_ip_dist;
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
template class distances_config_t<int, float>;

};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
