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

#include "common.h"
#include "coo_spmv_kernel.cuh"

namespace raft {
namespace sparse {
namespace distance {


template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy {

public:

    using smem_type = value_t;

    dense_smem_strategy(const distances_config_t<value_idx, value_t> &config_, int &smem_):
        config(config_),
        smem(smem_) { }

    /**
     * Computes the maximum number of columns that can be stored
     * in shared memory in dense form with the given block size
     * and precision.
     * @return the maximum number of columns that can be stored in smem
     */
    inline static int max_cols_per_block() {
        // max cols = (total smem available - offsets for A - cub reduction smem)
        return (raft::getSharedMemPerBlock() -
                ((tpb / raft::warp_size()) * sizeof(value_t))) /
                sizeof(value_t);
    }

    inline static int smem_per_block(int n_cols) {
        int max_cols = max_cols_per_block();
        if (n_cols > max_cols) {
            return -1;
        }
        return (n_cols * sizeof(value_t)) +
                ((tpb / raft::warp_size()) * sizeof(value_t));
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch(value_t *out_dists, value_idx *coo_rows_b, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        n_blocks_per_row = raft::ceildiv(config.b_nnz, chunk_size * tpb);
        n_blocks = config.a_nrows * n_blocks_per_row;

        CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
        CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
        CUML_LOG_DEBUG("smem_per_block: %d", smem);

        CUDA_CHECK(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<dense_smem_strategy, value_idx, value_t, tpb,
                                            false, product_f, accum_f,
                                            write_f>,cudaFuncCachePreferShared));

        balanced_coo_generalized_spmv_kernel<dense_smem_strategy, value_idx, value_t, tpb,
                                    false><<<n_blocks, tpb, smem, config.stream>>>( *this, config.a_indptr, config.a_indices, config.a_data, coo_rows_b,
        config.b_indices, config.b_data, config.a_nrows, config.b_nrows,
        config.b_ncols, config.b_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        n_blocks_per_row =  raft::ceildiv(config.a_nnz, chunk_size * tpb);
        n_blocks = config.b_nrows * n_blocks_per_row;

        CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
        CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
        CUML_LOG_DEBUG("smem_per_block: %d", smem);

        CUDA_CHECK(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<dense_smem_strategy, value_idx, value_t, tpb,
                                            true, product_f, accum_f,
                                            write_f>,cudaFuncCachePreferShared));

        balanced_coo_generalized_spmv_kernel<dense_smem_strategy, value_idx, value_t, tpb,
                                    true><<<n_blocks, tpb, smem, config.stream>>>( *this, config.b_indptr, config.b_indices, config.b_data, coo_rows_a,
        config.a_indices, config.a_data, config.b_nrows, config.a_nrows,
        config.a_ncols, config.a_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
    }

    __device__ inline void init_smem(smem_type *cache, value_idx &cache_size) {
        for (int k = threadIdx.x; k < cache_size; k += blockDim.x) cache[k] = 0.0;
    }

    __device__ inline void insert(smem_type *cache, value_idx &key, value_t &value) {
        cache[key] = value;
    }

    __device__ inline value_t find(smem_type *cache, value_idx &key) {
        return cache[key];
    }

private:
    int &smem, n_blocks_per_row, n_blocks;
    const distances_config_t<value_idx, value_t> &config;

};

} // namespace distance
} // namespace sparse
} // namespace raft