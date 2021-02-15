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

#include <cuml/common/logger.hpp>

#include <cuco/static_map.cuh>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx, typename value_t, int tpb>
class coo_spmv_strategy {

public:
    coo_spmv_strategy(const distances_config_t<value_idx, value_t> &config_) :
        config(config_) { }

    template <typename strategy_t, typename product_f, typename accum_f, typename write_f>
    void _dispatch_base(strategy_t &strategy, value_idx smem_dim, value_t *out_dists, value_idx *coo_rows_b, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
        CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
        CUML_LOG_DEBUG("smem_per_block: %d", smem);

        CUDA_CHECK(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<strategy_t, value_idx, value_t, tpb,
                                            false, product_f, accum_f,
                                            write_f>,cudaFuncCachePreferShared));

        balanced_coo_generalized_spmv_kernel<strategy_t, value_idx, value_t, tpb,
                                    false><<<n_blocks, tpb, smem, config.stream>>>(strategy, config.a_indptr, config.a_indices, config.a_data, coo_rows_b,
        config.b_indices, config.b_data, config.a_nrows, config.b_nrows,
        smem_dim, config.b_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
    }

    template <typename strategy_t, typename product_f, typename accum_f, typename write_f>
    void _dispatch_base_rev(strategy_t &strategy, value_idx smem_dim, value_t *out_dists, value_idx *coo_rows_a, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        CUML_LOG_DEBUG("n_blocks: %d", n_blocks);
        CUML_LOG_DEBUG("n_warps_per_row: %d", n_blocks_per_row);
        CUML_LOG_DEBUG("smem_per_block: %d", smem);

        CUDA_CHECK(cudaFuncSetCacheConfig(balanced_coo_generalized_spmv_kernel<strategy_t, value_idx, value_t, tpb,
                                            true, product_f, accum_f,
                                            write_f>,cudaFuncCachePreferShared));

        balanced_coo_generalized_spmv_kernel<strategy_t, value_idx, value_t, tpb,
                                    true><<<n_blocks, tpb, smem, config.stream>>>(strategy, config.b_indptr, config.b_indices, config.b_data, coo_rows_a,
        config.a_indices, config.a_data, config.b_nrows, config.a_nrows,
        smem_dim, config.a_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
    }

protected:
    int smem, n_blocks_per_row, n_blocks;
    const distances_config_t<value_idx, value_t> &config;
};

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block() {
    // max cols = (total smem available - cub reduction smem)
    return (raft::getSharedMemPerBlock() -
            ((tpb / raft::warp_size()) * sizeof(value_t))) /
            sizeof(value_t);
}

template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {

public:

    using smem_type = value_t*;
    using insert_type = smem_type;
    using find_type = smem_type;

    dense_smem_strategy(const distances_config_t<value_idx, value_t> &config_, int &smem_):
        coo_spmv_strategy<value_idx, value_t, tpb>(config_) {
            this->smem = smem_;
        }

    inline static int smem_per_block(int n_cols) {
        int max_cols = max_cols_per_block<value_idx, value_t, tpb>();
        if (n_cols > max_cols) {
            return -1;
        }
        return (n_cols * sizeof(value_t)) +
                ((tpb / raft::warp_size()) * sizeof(value_t));
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch(value_t *out_dists, value_idx *coo_rows_b, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        this-> n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
        this->n_blocks = this->config.a_nrows * this->n_blocks_per_row;

        this->_dispatch_base(*this, this->config.b_ncols, out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        this->n_blocks_per_row =  raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
        this->n_blocks = this->config.b_nrows * this->n_blocks_per_row;

        this->_dispatch_base_rev(*this, this->config.a_ncols, out_dists, coo_rows_a, product_func, accum_func, write_func, chunk_size);
    }

    __device__ inline insert_type init_insert(smem_type cache, value_idx &cache_size) {
        for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
            cache[k] = 0.0;
        }
        return cache;
    }

    __device__ inline void insert(insert_type cache, value_idx &key, value_t &value) {
        cache[key] = value;
    }

    __device__ inline find_type init_find(smem_type cache) {
        return cache;
    }

    __device__ inline value_t find(find_type cache, value_idx &key) {
        return cache[key];
    }

};

template <typename value_idx, typename value_t, int tpb>
class hash_strategy : public coo_spmv_strategy<value_idx, value_t, tpb>  {

public:

    // namespace cg = cooperative_groups;
    using insert_type = typename cuco::static_map<value_idx, value_t, cuda::thread_scope_block>::device_mutable_view;
    using smem_type = typename insert_type::slot_type*;
    using find_type = typename cuco::static_map<value_idx, value_t, cuda::thread_scope_block>::device_view;

    hash_strategy(const distances_config_t<value_idx, value_t> &config_):
        coo_spmv_strategy<value_idx, value_t, tpb>(config_) { 
            this->smem = raft::getSharedMemPerBlock();
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch(value_t *out_dists, value_idx *coo_rows_b, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        this->n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
        this->n_blocks = this->config.a_nrows * this->n_blocks_per_row;
    
        this->_dispatch_base(*this, map_size(), out_dists, coo_rows_b, product_func, accum_func, write_func, chunk_size);
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a, product_f product_func, accum_f accum_func, write_f write_func, int chunk_size) {
        this->n_blocks_per_row =  raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
        this->n_blocks = this->config.b_nrows * this->n_blocks_per_row;

        this->_dispatch_base_rev(*this, map_size(), out_dists, coo_rows_a, product_func, accum_func, write_func, chunk_size);
    }

    __device__ inline insert_type init_insert(smem_type cache, value_idx &cache_size) {
        return insert_type::make_from_uninitialized_slots(cooperative_groups::this_thread_block(), cache, map_size(), -1, 0);
    }

    __device__ inline void insert(insert_type cache, value_idx &key, value_t &value) {
        auto success = cache.insert(thrust::make_pair(key, value));
    }

    __device__ inline find_type init_find(smem_type cache) {
        return find_type(cache, map_size(), -1, 0);
    }

    __device__ inline value_t find(find_type cache, value_idx &key) {
        auto a_pair = cache.find(key);

        value_t a_col = 0.0;
        if (a_pair != cache.end()) {
            a_col = a_pair->second;
        }
        return a_col;
    }

private:

    __host__ __device__ constexpr int map_size() {
        return (46000 - ((tpb / raft::warp_size()) * sizeof(value_t))) / sizeof(typename insert_type::slot_type);
    }

};

} // namespace distance
} // namespace sparse
} // namespace raft