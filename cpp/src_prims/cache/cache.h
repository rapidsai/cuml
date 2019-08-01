/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <cub/cub.cuh>
#include "cache_util.h"
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "cuML.hpp"
#include "ml_utils.h"

namespace MLCommon {
namespace Cache {

using namespace MLCommon;

/**
* @brief Associative cache with least recently used replacement policy.
*
* SW managed cache in device memory, for ML algos where we can trade memory
* access for computation. The two main functions of this class are the
* management of cache indices, and methods to retrieve/store data using the
* cache indices.
*
* The index management can be considered as a hash map<int, int>, where the int
* keys are the original vector indices that we want to store, and the values are
* the cache location of these vectors. The indices are hashed into a bucket
* whose size equals the associativity. These are the cache sets. If a cache
* set is full, then new indices are stored by replacing the oldest entries.
*
* Using this index mapping we implement methods to store and retrive data from
* the cache buffer, where a unit of data that we are storing is math_t[n_vec].
* For example in SVM we store full columns of the kernel matrix at each cache
* entry.
*
* Note: we should have a look if the index management could be simplified using
* concurrent_unordered_map.cuh from cudf. See Issue #914.
*
* Example usage:
* @code{.cpp}
*
* // An expensive calculation that we want to accelerate with caching:
* // we have n keys, and for each key we generate a vector with m elements.
* // The keys and the output values are stored in GPU memory.
* void calc(int *key, int n, int m, float *out, cudaStream_t stream) {
*   for (k=0; k<n; k++) {
*     // use key[k] to generate out[i + m*k],  where i=0..m-1
*   }
* }
*
* // We assume that our ML algo repeatedly calls calc, and the set of keys have
* // an overlap. We will use the cache to avoid repeated calculations.
*
* // Assume we have cumlHandle_impl& h, and cudaStream_t stream
* Cache<float> cache(h.getDeviceAllocator(), stream, m);
*
* // A buffer that we will reuse to store the cache indices.
* device_buffer<int> cache_idx(h.getDeviceAllocator(), stream, n);
*
* void cached_calc(int *key, int n, int m, float *out, stream) {
*   int n_cached = 0;
*
*   cache.GetCacheIdxPartitioned(key, n, cache_idx.data(), &n_cached,
*                                cudaStream_t stream);
*
*   // Note: GetCacheIdxPartitioned has reordered the keys so that
*   // key[0..n_cached-1] are the keys already in the cache.
*   // We collect the corresponding values
*   cache.GetCols(cache_idx.data(), n_cached, out, stream);
*
*   // Calculate the elements not in the cache
*   int non_cached = n - n_cached;
*   if (non_cached > 0) {
*     int *key_new = key + n_cached;
*     int *cache_idx_new = cache_idx.data() + n_cached;
*     float *out_new = out + n_cached * m;
*     // AssignCacheIdx can permute the keys, therefore it has to come before
*     // we call calc.
*     // Note: a call to AssignCacheIdx should always be preceded with
*     // GetCacheIdxPartitioned, because that initializes the cache_idx_new array
*     // with the cache set (hash bucket) that correspond to the keys.
*     // The cache idx will be assigned from that cache set.
*     cache.AssignCacheIdx(key_new, non_cached, cache_idx_new, stream);
*
*     calc(key_new, non_cached, m, out_new, stream);
*
*     // Store the calculated vectors into the cache.
*     cache.StoreCols(out_new, non_cached, non_cached, cache_idx_new, stream);
*    }
* }
* @endcode
*/
template <typename math_t, int associativity = 32>
class Cache {
 public:
  /**
   * @brief Construct a Cache object
   *
   * @tparam math_t type of elements to be cached
   * @tparam associativity number of vectors in a cache set
   *
   * @param allocator device memory allocator
   * @param stream cuda stream
   * @param n_vec number of elements in a single vector that is stored in a
   *   cache entry
   * @param cache_size in MiB
   */
  Cache(std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
        int n_vec, float cache_size = 200)
    : allocator(allocator),
      n_vec(n_vec),
      cache_size(cache_size),
      cache(allocator, stream),
      vec_idx(allocator, stream),
      cache_time(allocator, stream),
      is_cached(allocator, stream),
      ws_tmp(allocator, stream),
      idx_tmp(allocator, stream),
      d_num_selected_out(allocator, stream, 1),
      d_temp_storage(allocator, stream) {
    ASSERT(n_vec > 0, "Parameter n_vec: shall be larger than zero");
    ASSERT(associativity > 0, "Associativity shall be larger than zero");
    ASSERT(cache_size >= 0, "Cache size should not be negative");

    // Calculate how many vectors would fit the cache
    int n_cache_vecs = (cache_size * 1024 * 1024) / (sizeof(math_t) * n_vec);

    // The available memory shall be enough for at least one cache set
    if (n_cache_vecs >= associativity) {
      n_cache_sets = n_cache_vecs / associativity;
      n_cache_vecs = n_cache_sets * associativity;
      cache.resize(n_cache_vecs * n_vec, stream);
      vec_idx.resize(n_cache_vecs, stream);
      cache_time.resize(n_cache_vecs, stream);
      CUDA_CHECK(cudaMemsetAsync(vec_idx.data(), 0,
                                 vec_idx.size() * sizeof(int), stream));
      CUDA_CHECK(cudaMemsetAsync(cache_time.data(), 0,
                                 cache_time.size() * sizeof(int), stream));
    } else {
      if (cache_size > 0) {
        std::cout << "Warning: not enough memory to cache a single set of "
                     "rows, not using cache\n";
      }
      n_cache_sets = 0;
      cache_size = 0;
    }
    //std::cout << "Creating cache with size " << cache_size << " MiB, to store " <<
    //  n_cache_vecs<< " vectors, in "<<n_cache_sets<<" sets with associativity "<<associativity<<"\n";
  }

  Cache(const Cache &other) = delete;

  Cache &operator=(const Cache &other) = delete;

  /** @brief Collect cached data into columns of contiguous memory space (using
   * column major memory layout).
   *
   * On exit, the tile array is filled the following way:
   * out[i + n_vec*k] = cache[i + n_vec * vec_idx[k]]), where i=0..n_vec-1,
   * k = 0..n-1
   *
   * @param [in] idx cache indices size [n]
   * @param [in] n the number of vectors that need to be collected
   * @param [out] out vectors collected from cache in column major format,
   *  size [n_vec*n]
   * @param [in] stream cuda stream
   */
  void GetCols(const int *idx, int n, math_t *out, cudaStream_t stream) {
    if (n > 0) {
      get_cols<<<ceildiv(n * n_vec, TPB), TPB, 0, stream>>>(cache.data(), n_vec,
                                                            idx, n, out);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  /** @brief Store column major data into the cache.
 * Roughly the opposite of GetCols, but the input columns can be scattered
 * in memory. The cache is updated using the following formula:
 *
 * cache[i + cache_idx[k]*n_vec] = tile[i + tile_idx[k]*n_vec],
 * for i=0..n_vec-1, k=0..n-1
 *
 * if tile_idx==nullptr, then we assume tile_idx[k] = k
 *
 * @param [in] tile stores the data to be cashed cached in column major format,
 *   size [n_vec x n_tile]
 * @param [in] n_tile number of columns in tile (at least n)
 * @param [in] n number of vectors that need to be stored in the cache (a subset
     of all the vectors in the tile)
 * @param [in] cache_idx cache indices for storing the vectors (negative values
 *   are ignored), size [n]
 * @param [in] stream cuda stream
 * @param [in] tile_idx indices of vectors that need to be stored
 */
  void StoreCols(const math_t *tile, int n_tile, int n, int *cache_idx,
                 cudaStream_t stream, const int *tile_idx = nullptr) {
    if (n > 0) {
      store_cols<<<ceildiv(n * n_vec, TPB), TPB, 0, stream>>>(
        tile, n_tile, n_vec, tile_idx, n, cache_idx, cache.data(),
        cache.size() / n_vec);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  /** @brief Map a set of indices to cache indices.
   *
   * For each k in 0..n-1, if in_idx[k] is found in the cache, then out_idx[k]
   * will tell the corresponding cache idx, and is_cached[k] is set to true.
   *
   * If in_idx[k] is not found in the cache, then is_cached[k] is set to false.
   * In this case we assign the cache set for in_idx[k], and out_idx[k] will
   * store the cache set.
   *
   * @Note in order to retrieve the cached vector j=out_idx[k] from the cache,
   *  we have to access cache[i + j*n_vec], where i=0..n_vec-1.
   *
   * @Note: do not use simultaneous GetCacheIdx and AssignCacheIdx
   *
   * @param [in] in_idx device array of column vector indices size [n]
   * @param [in] n number of indices
   * @param [out] out_idx device array of cache indices corresponding to the
   *   input idx size [n]
   * @param [out] is_cached whether the element is already available in the
   *   cache size [n]
   * @param [in] stream
   */
  void GetCacheIdx(int *in_idx, int n, int *out_idx, bool *is_cached,
                   cudaStream_t stream) {
    n_iter++;  // we increase the iteration counter, that is used to time stamp
    // accessing entries from the cache
    get_cache_idx<<<ceildiv(n, TPB), TPB, 0, stream>>>(
      in_idx, n, vec_idx.data(), n_cache_sets, associativity, cache_time.data(),
      out_idx, is_cached, n_iter);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  /** @brief Map a set of indices to cache indices.
   *
   * Same as GetCacheIdx, but partitions the in_idx, and out_idx arrays in a way
   * that in_idx[0..n_cached-1] and out_idx[0..n_cached-1] store the indices of
   * vectors that are found in the cache, while in_idx[n_cached..n-1] are the
   * indices of vectors that are not found in the cache. For the vectors not
   * found in the cache, out_idx[n_cached..n-1] stores the cache set, and this
   * can be used to call AssignCacheIdx.
   *
   * @param [inout] in_idx device array of column vector indices size [n]
   * @param [in] n number of indices
   * @param [out] out_idx device array of cache columns indices corresponding to
   *   the input idx size [n]
   * @param [out] n_cached number of elements that are cached
   * @param [in] stream
   */
  void GetCacheIdxPartitioned(int *in_idx, int n, int *out_idx, int *n_cached,
                              cudaStream_t stream) {
    ResizeTmpBuffers(n, stream);

    GetCacheIdx(in_idx, n, ws_tmp.data(), is_cached.data(), stream);

    // Group cache indices as [already cached, non_cached]
    cub::DevicePartition::Flagged(d_temp_storage.data(), d_temp_storage_size,
                                  ws_tmp.data(), is_cached.data(), out_idx,
                                  d_num_selected_out.data(), n, stream);

    updateHost(n_cached, d_num_selected_out.data(), 1, stream);

    // Similarily re-group the input indices
    copy(ws_tmp.data(), in_idx, n, stream);
    cub::DevicePartition::Flagged(d_temp_storage.data(), d_temp_storage_size,
                                  ws_tmp.data(), is_cached.data(), in_idx,
                                  d_num_selected_out.data(), n, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  /**
   * @brief Assign cache location to a set of indices.
   *
   * Note: calle GetCacheIdx first, to get the cache_set assigned to the input
   * vectors.
   *
   * @param [inout] idx  device array of vector indices size [n]
   * @param [in] n number of elements that we want to cache
   * @param [inout] cidx on entry: cache_set, on exit: assigned cache_idx, or -1
   *   size[n]
   * @param [in] stream cuda stream
   */
  void AssignCacheIdx(int *idx, int n, int *cidx, cudaStream_t stream) {
    if (n <= 0) return;
    cub::DeviceRadixSort::SortPairs(d_temp_storage.data(), d_temp_storage_size,
                                    cidx, ws_tmp.data(), idx, idx_tmp.data(), n,
                                    0, sizeof(int) * 8, stream);

    copy(idx, idx_tmp.data(), n, stream);

    // set it to -1
    CUDA_CHECK(cudaMemsetAsync(cidx, 255, n * sizeof(int), stream));
    const int nthreads = associativity <= 32 ? associativity : 32;

    assign_cache_idx<nthreads, associativity>
      <<<n_cache_sets, nthreads, 0, stream>>>(idx, n, ws_tmp.data(),
                                              vec_idx.data(), n_cache_sets,
                                              cache_time.data(), n_iter, cidx);

    CUDA_CHECK(cudaPeekAtLastError());
    if (debug_mode) CUDA_CHECK(cudaDeviceSynchronize());
  }

  /** Return approximate cache size in MiB. */
  float GetSizeInMiB() const { return cache_size; }

  /**
   * Returns the number of vectors that can be cached.
   */
  int GetSize() const { return vec_idx.size(); }

 private:
  std::shared_ptr<deviceAllocator> allocator;

  int n_vec;         //!< Number of elements in a cached vector
  float cache_size;  //!< in MiB
  int n_cache_sets;  //!< number of cache sets

  const int TPB = 256;  //!< threads per block for kernel launch
  int n_iter = 0;       //!< Counter for time stamping cache operation

  bool debug_mode = false;

  MLCommon::device_buffer<math_t> cache;    //!< The value of cached vectors
  MLCommon::device_buffer<int> vec_idx;     //!< Indices of vectors stored
  MLCommon::device_buffer<int> cache_time;  //!< Time stamp for LRU cache

  // Helper arrays for GetCacheIdx
  MLCommon::device_buffer<bool> is_cached;
  MLCommon::device_buffer<int> ws_tmp;
  MLCommon::device_buffer<int> idx_tmp;

  // Helper arrays for cub
  MLCommon::device_buffer<int> d_num_selected_out;
  MLCommon::device_buffer<char> d_temp_storage;
  size_t d_temp_storage_size = 0;

  void ResizeTmpBuffers(int n, cudaStream_t stream) {
    if (ws_tmp.size() < n) {
      ws_tmp.resize(n, stream);
      is_cached.resize(n, stream);
      idx_tmp.resize(n, stream);
      cub::DevicePartition::Flagged(NULL, d_temp_storage_size, vec_idx.data(),
                                    is_cached.data(), vec_idx.data(),
                                    d_num_selected_out.data(), n, stream);
      d_temp_storage.resize(d_temp_storage_size, stream);
    }
  }
};

};  // namespace Cache
};  // namespace MLCommon
