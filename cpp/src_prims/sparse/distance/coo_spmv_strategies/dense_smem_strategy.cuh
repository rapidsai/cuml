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

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx, typename value_t, int tpb>
class dense_smem_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  using smem_type = value_t *;
  using insert_type = smem_type;
  using find_type = smem_type;

  dense_smem_strategy(const distances_config_t<value_idx, value_t> &config_,
                      int &smem_)
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_) {
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
  void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                product_f product_func, accum_f accum_func, write_f write_func,
                int chunk_size) {
    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
    auto n_blocks = this->config.a_nrows * n_blocks_per_row;

    mask_row_it<value_idx> a_indptr(this->config.a_indptr,
                                    this->config.a_nrows);

    this->_dispatch_base(*this, this->config.b_ncols, a_indptr, out_dists,
                         coo_rows_b, product_func, accum_func, write_func,
                         chunk_size, n_blocks, n_blocks_per_row);
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a,
                    product_f product_func, accum_f accum_func,
                    write_f write_func, int chunk_size) {
    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
    auto n_blocks = this->config.b_nrows * n_blocks_per_row;

    mask_row_it<value_idx> b_indptr(this->config.b_indptr,
                                    this->config.b_nrows);

    this->_dispatch_base_rev(*this, this->config.a_ncols, b_indptr, out_dists,
                             coo_rows_a, product_func, accum_func, write_func,
                             chunk_size, n_blocks, n_blocks_per_row);
  }

  __device__ inline insert_type init_insert(smem_type cache,
                                            value_idx &cache_size) {
    for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
      cache[k] = 0.0;
    }
    return cache;
  }

  __device__ inline void insert(insert_type cache, value_idx &key,
                                value_t &value) {
    cache[key] = value;
  }

  __device__ inline find_type init_find(smem_type cache) { return cache; }

  __device__ inline value_t find(find_type cache, value_idx &key, value_idx *indices, value_t *data, value_idx start_offset, value_idx stop_offset) {
    return cache[key];
  }
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft