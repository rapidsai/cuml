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
#include "utils.cuh"

#include <cuml/common/logger.hpp>

#include <cuco/static_map.cuh>

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
        strategy, a_indptr, config.a_indices, config.a_data, coo_rows_b,
        config.b_indices, config.b_data, a_indptr.n_rows, config.b_nrows,
        smem_dim, config.b_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
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
        strategy, b_indptr, config.b_indices, config.b_data, coo_rows_a,
        config.a_indices, config.a_data, b_indptr.n_rows, config.a_nrows,
        smem_dim, config.a_nnz, out_dists, n_blocks_per_row, chunk_size,
        product_func, accum_func, write_func);
  }

 protected:
  int smem;
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

template <typename value_idx>
class mask_indptr_it {
 public:
  mask_indptr_it(const value_idx *full_indptr_, const value_idx &n_rows_,
                 value_idx *mask_row_idx_ = NULL)
    : full_indptr(full_indptr_), mask_row_idx(mask_row_idx_), n_rows(n_rows_) {}

  __device__ inline value_idx get_row_idx(const int &n_blocks_nnz_b) {
    if (mask_row_idx != NULL) {
      return mask_row_idx[blockIdx.x / n_blocks_nnz_b];
    } else {
      return blockIdx.x / n_blocks_nnz_b;
    }
  }

  __device__ inline void get_row_offsets(const value_idx &row_idx,
                                         value_idx &start_offset,
                                         value_idx &stop_offset,
                                         const value_idx &n_blocks_nnz_b) {
    start_offset = full_indptr[row_idx];
    stop_offset = full_indptr[row_idx + 1];
  }

  const value_idx *full_indptr, &n_rows;
  value_idx *mask_row_idx;
};

template <typename value_idx>
__global__ void fill_chunk_indices_kernel(value_idx *n_chunks_per_row,
                                          value_idx *chunk_indices,
                                          value_idx n_rows) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_rows) {
    auto start = n_chunks_per_row[tid];
    auto end = n_chunks_per_row[tid + 1];

// auto row_idx = mask_row_idx[tid];
#pragma unroll
    for (int i = start; i < end; i++) {
      chunk_indices[i] = tid;
    }
  }
}

template <typename value_idx>
class chunked_mask_indptr_it : public mask_indptr_it<value_idx> {
 public:
  chunked_mask_indptr_it(const value_idx *full_indptr_,
                         const value_idx &n_rows_, value_idx *mask_row_idx_,
                         int row_chunk_size_, const cudaStream_t stream_)
    : mask_indptr_it<value_idx>(full_indptr_, n_rows_, mask_row_idx_),
      row_chunk_size(row_chunk_size_),
      stream(stream_) {}

  void init() {
    auto policy = rmm::exec_policy(stream);
    CUDA_CHECK(cudaMalloc(&row_chunk_size_d, 1 * sizeof(int)));
    raft::update_device(row_chunk_size_d, &row_chunk_size, 1, stream);

    // set first element as 0, and rest are row indices from mask
    n_chunks_per_row = rmm::device_vector<value_idx>(this->n_rows + 1);
    CUDA_CHECK(cudaMemsetAsync(n_chunks_per_row.data().get(), 0,
                               sizeof(value_idx) * 1, stream));
    n_chunks_per_row_functor chunk_functor(this->full_indptr, row_chunk_size_d);
    thrust::transform(policy, this->mask_row_idx,
                      this->mask_row_idx + this->n_rows,
                      n_chunks_per_row.begin() + 1, chunk_functor);

    thrust::inclusive_scan(policy, n_chunks_per_row.begin() + 1,
                           n_chunks_per_row.end(),
                           n_chunks_per_row.begin() + 1);

    n_chunks_per_row_ptr = n_chunks_per_row.data().get();
    raft::update_host(&total_row_blocks, n_chunks_per_row_ptr + this->n_rows, 1,
                      stream);

    fill_chunk_indices();
  }

  __device__ inline value_idx get_row_idx(const int &n_blocks_nnz_b) {
    return this->mask_row_idx[chunk_indices_ptr[blockIdx.x / n_blocks_nnz_b]];
  }

  __device__ inline void get_row_offsets(const value_idx &row_idx,
                                         value_idx &start_offset,
                                         value_idx &stop_offset,
                                         const int &n_blocks_nnz_b) {
    auto chunk_index = blockIdx.x / n_blocks_nnz_b;
    auto chunk_val = chunk_indices_ptr[chunk_index];
    auto prev_n_chunks = n_chunks_per_row_ptr[chunk_val];
    auto relative_chunk = chunk_index - prev_n_chunks;

    start_offset = relative_chunk * row_chunk_size_d[0];
    stop_offset = start_offset + row_chunk_size_d[0];

    auto final_stop_offset = this->full_indptr[row_idx + 1];

    stop_offset =
      stop_offset > final_stop_offset ? final_stop_offset : stop_offset;
  }

  value_idx total_row_blocks;

  const cudaStream_t stream;
  rmm::device_vector<value_idx> n_chunks_per_row, chunk_indices;
  value_idx *n_chunks_per_row_ptr, *chunk_indices_ptr;
  int &row_chunk_size, *row_chunk_size_d;

  struct n_chunks_per_row_functor {
   public:
    n_chunks_per_row_functor(const value_idx *indptr_, int *row_chunk_size_)
      : indptr(indptr_), row_chunk_size(row_chunk_size_) {}

    __host__ __device__ value_idx operator()(const value_idx &i) {
      auto degree = indptr[i + 1] - indptr[i];
      return raft::ceildiv(degree, (value_idx)row_chunk_size[0]);
    }

    const value_idx *indptr;
    int *row_chunk_size;
  };

  void fill_chunk_indices() {
    auto n_threads = std::min(this->n_rows, 256);
    auto n_blocks = raft::ceildiv(this->n_rows, (value_idx)n_threads);

    chunk_indices = rmm::device_vector<value_idx>(total_row_blocks);
    // chunk_indices.resize(total_row_blocks, stream);
    chunk_indices_ptr = chunk_indices.data().get();

    fill_chunk_indices_kernel<value_idx><<<n_blocks, n_threads, 0, stream>>>(
      n_chunks_per_row_ptr, chunk_indices_ptr, this->n_rows);
  }
};

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

    mask_indptr_it<value_idx> a_indptr(this->config.a_indptr,
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

    mask_indptr_it<value_idx> b_indptr(this->config.b_indptr,
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

  __device__ inline value_t find(find_type cache, value_idx &key) {
    return cache[key];
  }
};

template <typename value_idx, typename value_t, int tpb>
class hash_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  // namespace cg = cooperative_groups;
  using insert_type =
    typename cuco::static_map<value_idx, value_t,
                              cuda::thread_scope_block>::device_mutable_view;
  using smem_type = typename insert_type::slot_type *;
  using find_type =
    typename cuco::static_map<value_idx, value_t,
                              cuda::thread_scope_block>::device_view;

  hash_strategy(const distances_config_t<value_idx, value_t> &config_)
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_), mask_indptr(1) {
    this->smem = raft::getSharedMemPerBlock();
  }

  bool chunking_needed(const value_idx *indptr, const value_idx n_rows) {
    auto widest_row =
      max_degree<value_idx, true>(indptr, n_rows, this->config.allocator,
                                  this->config.stream, 0.5 * map_size());

    // figure out if chunking strategy needs to be enabled
    // operating at 50% of hash table size
    if (widest_row.first > 0.5 * map_size()) {
      chunking = true;
      more_rows = widest_row.second;
      less_rows = n_rows - more_rows;
      mask_indptr = rmm::device_vector<value_idx>(n_rows);

      fits_in_hash_table<true> fits_functor(indptr);
      thrust::copy_if(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_rows),
                      mask_indptr.data().get(), fits_functor);
      fits_in_hash_table<false> not_fits_functor(indptr);
      thrust::copy_if(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(n_rows),
                      mask_indptr.data().get() + less_rows, not_fits_functor);
    } else {
      chunking = false;
    }
    return chunking;
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                product_f product_func, accum_f accum_func, write_f write_func,
                int chunk_size) {
    auto need = chunking_needed(this->config.a_indptr, this->config.a_nrows);

    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);

    if (need) {
      mask_indptr_it<value_idx> less(this->config.a_indptr, less_rows,
                                     mask_indptr.data().get());
      chunked_mask_indptr_it<value_idx> more(
        this->config.a_indptr, more_rows, mask_indptr.data().get() + less_rows,
        0.5 * map_size(), this->config.stream);
      more.init();

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base(*this, map_size(), less, out_dists, coo_rows_b,
                           product_func, accum_func, write_func, chunk_size,
                           n_less_blocks, n_blocks_per_row);

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base(*this, map_size(), more, out_dists, coo_rows_b,
                           product_func, accum_func, write_func, chunk_size,
                           n_more_blocks, n_blocks_per_row);
    } else {
      mask_indptr_it<value_idx> less(this->config.a_indptr,
                                     this->config.a_nrows);

      auto n_blocks = this->config.a_nrows * n_blocks_per_row;
      this->_dispatch_base(*this, map_size(), less, out_dists, coo_rows_b,
                           product_func, accum_func, write_func, chunk_size,
                           n_blocks, n_blocks_per_row);
    }
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a,
                    product_f product_func, accum_f accum_func,
                    write_f write_func, int chunk_size) {
    auto need = chunking_needed(this->config.b_indptr, this->config.b_nrows);

    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);

    if (need) {
      mask_indptr_it<value_idx> less(this->config.b_indptr, less_rows,
                                     mask_indptr.data().get());
      chunked_mask_indptr_it<value_idx> more(
        this->config.b_indptr, more_rows, mask_indptr.data().get() + less_rows,
        0.5 * map_size(), this->config.stream);
      more.init();

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base_rev(*this, map_size(), less, out_dists, coo_rows_a,
                               product_func, accum_func, write_func, chunk_size,
                               n_less_blocks, n_blocks_per_row);

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base_rev(*this, map_size(), more, out_dists, coo_rows_a,
                               product_func, accum_func, write_func, chunk_size,
                               n_more_blocks, n_blocks_per_row);
    } else {
      mask_indptr_it<value_idx> less(this->config.b_indptr,
                                     this->config.b_nrows);

      auto n_blocks = this->config.a_nrows * n_blocks_per_row;
      this->_dispatch_base_rev(*this, map_size(), less, out_dists, coo_rows_a,
                               product_func, accum_func, write_func, chunk_size,
                               n_blocks, n_blocks_per_row);
    }
  }

  __device__ inline insert_type init_insert(smem_type cache,
                                            value_idx &cache_size) {
    return insert_type::make_from_uninitialized_slots(
      cooperative_groups::this_thread_block(), cache, map_size(), -1, 0);
  }

  __device__ inline void insert(insert_type cache, value_idx &key,
                                value_t &value) {
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
  __host__ __device__ constexpr static int map_size() {
    return (48000 - ((tpb / raft::warp_size()) * sizeof(value_t))) /
           sizeof(typename insert_type::slot_type);
  }

  bool chunking = false;
  value_idx less_rows, more_rows;
  rmm::device_vector<value_idx> mask_indptr;

  template <bool fits>
  struct fits_in_hash_table {
    fits_in_hash_table(const value_idx *indptr_) : indptr(indptr_) {}

    __host__ __device__ bool operator()(const value_idx &i) {
      auto degree = indptr[i + 1] - indptr[i];

      if (fits) {
        return degree <= 0.5 * hash_strategy::map_size();
      } else {
        return degree > 0.5 * hash_strategy::map_size();
      }
    }

   private:
    const value_idx *indptr;
  };
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft