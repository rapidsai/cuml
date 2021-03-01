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

#include "coo_mask_row_iterators.cuh"
#include "../common.h"
#include "../detail/coo_spmv_kernel.cuh"
#include "../utils.cuh"

#include <cuml/common/logger.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx, typename value_t, int tpb>
class coo_spmv_strategy {
 public:
  coo_spmv_strategy(const distances_config_t<value_idx, value_t> &config_)
    : config(config_) {}

  template <typename strategy_t, typename indptr_it, typename product_f,
            typename accum_f, typename write_f>
  void _dispatch_base(strategy_t &strategy, value_idx smem_dim,
                      indptr_it &a_indptr, value_t *out_dists,
                      value_idx *coo_rows_b, product_f product_func,
                      accum_f accum_func, write_f write_func, int chunk_size,
                      int n_blocks, int n_blocks_per_row) {
    CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
    CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
    CUML_LOG_DEBUG("smem_per_block: %d", smem);

    CUDA_CHECK(cudaFuncSetCacheConfig(
      balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx,
                                           value_t, tpb, false, product_f,
                                           accum_f, write_f>,
      cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx,
                                         value_t, tpb, false>
      <<<n_blocks, tpb, smem, config.stream>>>(
        strategy, a_indptr, config.a_indices, config.a_data, config.a_nnz,
        coo_rows_b, config.b_indices, config.b_data, config.a_nrows,
        config.b_nrows, smem_dim, config.b_nnz, out_dists, n_blocks_per_row,
        chunk_size, product_func, accum_func, write_func);
  }

  template <typename strategy_t, typename indptr_it, typename product_f,
            typename accum_f, typename write_f>
  void _dispatch_base_rev(strategy_t &strategy, value_idx smem_dim,
                          indptr_it &b_indptr, value_t *out_dists,
                          value_idx *coo_rows_a, product_f product_func,
                          accum_f accum_func, write_f write_func,
                          int chunk_size, int n_blocks, int n_blocks_per_row) {
    CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
    CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
    CUML_LOG_DEBUG("smem_per_block: %d", smem);

    CUDA_CHECK(cudaFuncSetCacheConfig(
      balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx,
                                           value_t, tpb, true, product_f,
                                           accum_f, write_f>,
      cudaFuncCachePreferShared));

    balanced_coo_generalized_spmv_kernel<strategy_t, indptr_it, value_idx,
                                         value_t, tpb, true>
      <<<n_blocks, tpb, smem, config.stream>>>(
        strategy, b_indptr, config.b_indices, config.b_data, config.b_nnz,
        coo_rows_a, config.a_indices, config.a_data, config.b_nrows,
        config.a_nrows, smem_dim, config.a_nnz, out_dists, n_blocks_per_row,
        chunk_size, product_func, accum_func, write_func);
  }

 protected:
  int smem;
  const distances_config_t<value_idx, value_t> &config;
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft