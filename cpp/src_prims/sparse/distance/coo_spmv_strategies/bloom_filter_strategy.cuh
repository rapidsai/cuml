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

#include "base_strategy.cuh"

#include <cuco/detail/hash_functions.cuh>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx, typename value_t, int tpb>
class bloom_filter_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {

public:
    using smem_type = uint32_t *;
    using insert_type = smem_type;
    using find_type = smem_type;

    using Hash1 = cuco::detail::MurmurHash3_32<value_idx>;
    using Hash2 = cuco::detail::MurmurHash3_32<value_idx>;
    using Hash3 = cuco::detail::MurmurHash3_32<value_idx>;
  
    bloom_filter_strategy(const distances_config_t<value_idx, value_t> &config_, mask_row_it<value_idx> &row_it_)
      : coo_spmv_strategy<value_idx, value_t, tpb>(config_),
        row_it(row_it_),
        hash1(config_.a_nnz),
        hash2(config_.a_nrows),
        hash3(config_.a_ncols) {
      this->smem = raft::getSharedMemPerBlock();
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                  product_f product_func, accum_f accum_func, write_f write_func,
                  int chunk_size) {

        auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
        auto n_blocks = row_it.n_rows * n_blocks_per_row;

        this->_dispatch_base(*this, filter_size(), row_it, out_dists, coo_rows_b,
        product_func, accum_func, write_func, chunk_size,
        n_blocks, n_blocks_per_row);
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch_rev(value_t *out_dists, value_idx *coo_rows_b,
                product_f product_func, accum_f accum_func, write_f write_func,
                int chunk_size) {
        auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
        auto n_blocks = row_it.n_rows * n_blocks_per_row;

        this->_dispatch_base_rev(*this, filter_size(), row_it, out_dists, coo_rows_b,
        product_func, accum_func, write_func, chunk_size,
        n_blocks, n_blocks_per_row);
    }

    __device__ inline insert_type init_insert(smem_type cache,
        value_idx &cache_size) {
        for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
            cache[k] = 0.0;
        }
        return cache;
    }

    __device__ inline void _set_key(insert_type filter, uint32_t &h) {
        auto size = sizeof(uint32_t);
        uint32_t mem_idx = h;
        uint32_t mem_bit = size - (h % size);
        uint32_t val;
        uint32_t old;
        do {
          val = filter[mem_idx];
          old = atomicCAS(filter+mem_idx, val, val | 1 << mem_bit);
        } while(val != old);
    }

      __device__ inline void insert(insert_type filter, value_idx &key, value_t &value) {
        uint32_t hashed1 = hash1(key) & (filter_size() - 1);
        uint32_t hashed2 = hash2(key) & (filter_size() - 1);
        uint32_t hashed3 = hash3(key) & (filter_size() - 1);
        _set_key(filter, hashed1);
        _set_key(filter, hashed2);
        _set_key(filter, hashed3);
    }

    __device__ inline find_type init_find(smem_type cache) { return cache; }

    __device__ inline bool _get_key(find_type filter, uint32_t &h) {
        auto size = sizeof(uint32_t);
        uint32_t mem_idx = h;
        uint32_t mem_bit = size - (h % size);
        return (filter[mem_idx] & 1 << mem_bit) > 0;
    }

    __device__ inline value_t find(find_type filter, value_idx &key, value_idx *indices, value_t *data, value_idx start_offset, value_idx stop_offset) {
        uint32_t hashed1 = hash1(key) & (filter_size() - 1);
        uint32_t hashed2 = hash2(key) & (filter_size() - 1);
        uint32_t hashed3 = hash3(key) & (filter_size() - 1);
        /**
         * and 2? other hash functions would be useful
         */
        auto key_present = _get_key(filter, hashed1) && _get_key(filter, hashed2) &&
                      _get_key(filter, hashed3);
        // printf("index_b: %d, key_present: %d\n", key, key_present);
        if (!key_present) {
            return 0.0;
        }
        else {
            while (start_offset <= stop_offset) {
                value_idx mid = start_offset + (stop_offset - start_offset) / 2;

                auto mid_val = indices[mid];
                if (mid_val == key) {
                    return data[mid];
                }
                else if (mid_val < key) {
                    start_offset = mid + 1;
                }
                else if (mid_val > key) {
                    stop_offset = mid - 1;
                }
            }
            return 0.0;
        }
    }

private:
    __host__ __device__ constexpr static int filter_size() {
        return (48000 - ((tpb / raft::warp_size()) * sizeof(value_t))) /
               sizeof(uint32_t);
        // return 2;
    }

    Hash1 hash1;
    Hash2 hash2;
    Hash3 hash3;
    mask_row_it<value_idx> &row_it;
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft