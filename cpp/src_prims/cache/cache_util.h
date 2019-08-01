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
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "cuML.hpp"
#include "ml_utils.h"
#include "selection/kselection.h"

namespace MLCommon {
namespace Cache {

/**
 * @brief Collect vectors of data from the cache into a contiguous memory buffer.
 * 
 * We assume contiguous memory layout for the output buffer, i.e. we get
 * column vectors into a column major out buffer, or row vectors into a row 
 * major output buffer.
 *
 * On exit, the output array is filled the following way:
 * out[i + n_vec*k] = cache[i + n_vec * cache_idx[k]]), where i=0..n_vec-1, and
 *   k = 0..n-1
 *
 * @param [in] cache stores the cached data, size [n_vec x n_cached_vectors]
 * @param [in] n_vec number of elements in a cached vector
 * @param [in] cache_idx cache indices, size [n]
 * @param [in] n the number of elements that need to be collected
 * @param [out] out vectors collected from the cache, size [n_vec * n]
 */
template <typename math_t>
__global__ void get_vecs(const math_t *cache, int n_vec, const int *cache_idx,
                         int n, math_t *out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int row = tid % n_vec;  // row idx
  if (tid < n_vec * n) {
    int out_col = tid / n_vec;  // col idx
    int cache_col = cache_idx[out_col];
    if (row + out_col * n_vec < n_vec * n) {
      out[tid] = cache[row + cache_col * n_vec];
    }
  }
}

/**
 * @brief Store vectors of data into the cache. 
 * 
 * Elements within a vector should be contiguous in memory (i.e. column vectors 
 * for column major data storage, or row vectors of row major data). 
 *
 * If tile_idx==nullptr then the operation is the opposite of get_vecs, 
 * i.e. we store
 * cache[i + cache_idx[k]*n_vec] = tile[i + k*n_vec], for i=0..n_vec-1, k=0..n-1
 *
 * If tile_idx != nullptr, then  we permute the vectors from tile according
 * to tile_idx. This allows to store vectors from a buffer where the individual
 * vectors are not stored contiguously (but the elements of each vector shall 
 * be contiguous):
 * cache[i + cache_idx[k]*n_vec] = tile[i + tile_idx[k]*n_vec],
 * for i=0..n_vec-1, k=0..n-1
 *
 * @param [in] tile stores the data to be cashed cached, size [n_vec x n_tile]
 * @param [in] n_tile number of vectors in the input tile
 * @param [in] n_vec number of elements in a cached vector
 * @param [in] tile_idx indices of vectors that we want to store
 * @param [in] n number of vectos that we want to store (n <= n_tile)
 * @param [in] cache_idx cache indices, size [n], negative values are ignored
 * @param [inout] cache updated cache
 * @param [in] n_cache_vecs
 */
template <typename math_t>
__global__ void store_vecs(const math_t *tile, int n_tile, int n_vec,
                           const int *tile_idx, int n, const int *cache_idx,
                           math_t *cache, int n_cache_vecs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int row = tid % n_vec;  // row idx
  if (tid < n_vec * n) {
    int tile_col = tid / n_vec;  // col idx
    int data_col = tile_idx ? tile_idx[tile_col] : tile_col;
    int cache_col = cache_idx[tile_col];

    // We ignore negative values. The rest of the checks should be fulfilled
    // if the cache is used properly
    if (cache_col >= 0 && cache_col < n_cache_vecs && data_col < n_tile) {
      cache[row + cache_col * n_vec] = tile[row + data_col * n_vec];
    }
  }
}

/**
 * Map a key to a cache set.
 */
int DI hash(int key, int n_cache_sets, int associativity) {
  return key % n_cache_sets;
}

/**
 * @brief Binary search to find the first element in the array which is greater
 * equal than a given value.
 * @param [in] array sorted array of n numbers
 * @param [in] n length of the array
 * @param [in] value
 * @return the index of the first element in the array for which
 * array[idx] >= value. If there is no such value, then return n.
 */
int DI arg_first_ge(const int *array, int n, int val) {
  int start = 0;
  int end = n - 1;
  if (array[0] == val) return 0;
  if (array[end] < val) return n;
  while (start + 1 < end) {
    int q = (start + end + 1) / 2;
    //invariants:
    // start < end
    // start < q <=end
    // array[start] < val && array[end] <=val
    // at every iteration d = end-start is decreasing
    // when d==0, then array[end] will be the first element >= val.
    if (array[q] >= val) {
      end = q;
    } else {
      start = q;
    }
  }
  return end;
}
/**
 * @brief Find the k-th occurrence of value in a sorted array.
 *
 * Assume that array is [0, 1, 1, 1, 2, 2, 4, 4, 4, 4, 6, 7]
 * then find_nth_occurrence(cset, 12, 4, 2) == 7, because cset_array[7] stores
 * the second element with value = 4.
 * If there are less then k values in the array, then return -1
 *
 * @param [in] array sorted array of numbers, size [n]
 * @param [in] n number of elements in the array
 * @param [in] val the value we are searching for
 * @param [in] k
 * @return the idx of the k-th occurance of val in array, or -1 if
 * the value is not found.
 */
int DI find_nth_occurrence(const int *array, int n, int val, int k) {
  int q = arg_first_ge(array, n, val);
  if (q + k < n && array[q + k] == val) {
    q += k;
  } else {
    q = -1;
  }
  return q;
}

/**
 * @brief Rank the entries in a cache set according the time stamp, return the
 * indices that would sort the time stamp in ascending order.
 *
 * Assume we have a single cache set with time stamps as:
 * key (threadIdx.x):   0   1   2   3
 * val (time stamp):    8   6   7   5
 *
 * The corresponding sorted key-value pairs:
 * key:    3   1   2   0
 * val:    5   6   7   8
 * rank: 0th 1st 2nd 3rd
 *
 * On return, the rank is assigned or each thread:
 * threadIdx.x: 0   1   2   3
 * rank:        3   1   2   0
 *
 * For multiple cache sets, launch one block per cache set.
 *
 * @tparam nthreads number of threads per block (nthreads <= associativity)
 * @tparam associativity number of items in a cache set
 *
 * @param [in] time time stamp of caching the data,
     size [associativity * n_cache_sets]
 * @param [in] n_cache_sets number of cache sets
 * @param [out] rank within the cache set size [nthreads * items_per_thread]
 *   Each block should give a different pointer for rank.
 */
template <int nthreads, int associativity>
DI void rank_set_entries(const int *cache_time, int n_cache_sets, int *rank) {
  const int items_per_thread = ceildiv(associativity, nthreads);
  typedef cub::BlockRadixSort<int, nthreads, items_per_thread, int>
    BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int key[items_per_thread];
  int val[items_per_thread];

  int block_offset = blockIdx.x * associativity;

  for (int j = 0; j < items_per_thread; j++) {
    int k = threadIdx.x + j * nthreads;
    int t = (k < associativity) ? cache_time[block_offset + k] : 32768;
    key[j] = t;
    val[j] = k;
  }

  BlockRadixSort(temp_storage).Sort(key, val);

  for (int j = 0; j < items_per_thread; j++) {
    if (val[j] < associativity) {
      rank[val[j]] = threadIdx.x * items_per_thread + j;
    }
  }
  __syncthreads();
}

/**
 * @brief Assign cache location to a set of keys using LRU replacement policy.
 *
 * The keys and the corresponding cache_set arrays shall be sorted according
 * to cache_set in ascending order. One block should be launched for every cache
 * set.
 *
 * Each cache set is sorted according to time_stamp, and values from keys
 * are filled in starting at the oldest time stamp. Enties that were accessed
 * at the current time are not reassigned.
 *
 * @tparam nthreads number of threads per block
 * @tparam assaciativity number of keys in a cache set
 *
 * @param [in] keys that we want to cache size [n]
 * @param [in] n number of keys
 * @param [in] cache_set assigned to keys, size [n]
 * @param [inout] cached_keys keys of already cached vectors,
 *   size [n_cache_sets*associativity], on exit it will be updated with the
 *   cached elements from keys.
 * @param [in] n_cache_sets number of cache sets
 * @param [inout] cache_time will be updated to "time" for those elements that
 *   could be assigned to a cache location, size [n_cache_sets*associativity]
 * @param [in] time time stamp
 * @param [out] cache_idx the cache idx assigned to the input, or -1 if it could
 *   not be cached, size [n]
 */
template <int nthreads, int associativity>
__global__ void assign_cache_idx(const int *keys, int n, const int *cache_set,
                                 int *cached_keys, int n_cache_sets,
                                 int *cache_time, int time, int *cache_idx) {
  int block_offset = blockIdx.x * associativity;

  const int items_per_thread = ceildiv(associativity, nthreads);

  // the size of rank limits how large associativity can be used in practice
  __shared__ int rank[items_per_thread * nthreads];
  rank_set_entries<nthreads, associativity>(cache_time, n_cache_sets, rank);

  // Each thread will fill items_per_thread items in the cache.
  // It uses a place, only if it was not udated at the current time step
  // (cache_time != time).
  // We rank the places acconding to the time stamp, least recently used
  // elements come to the front.
  // We fill the least recently used elements with the working set.
  // there might be elements which cannot be assigned to cache loc.
  // these elements are assigned -1.

  for (int j = 0; j < items_per_thread; j++) {
    int i = threadIdx.x + j * nthreads;
    int t_idx = block_offset + i;
    bool mask = (i < associativity);
    // whether this slot is available for writing
    mask = mask && (cache_time[t_idx] != time);

    // rank[i] tells which element to store by this thread
    // we look up where is the corresponding key stored in the input array
    if (mask) {
      int k = find_nth_occurrence(cache_set, n, blockIdx.x, rank[i]);
      if (k > -1) {
        int key_val = keys[k];
        cached_keys[t_idx] = key_val;
        cache_idx[k] = t_idx;
        cache_time[t_idx] = time;
      }
    }
  }
}

/**
 * @brief Get the cache indices for keys stored in the cache.
 *
 * For every key, we look up the corresponding cache position.
 * If keys[k] is stored in the cache, then is_cached[k] is set to true, and
 * cache_idx[k] stores the corresponding cache idx.
 *
 * If keys[k] is not stored in the cache, then we assign a cache set to it.
 * This  cache set is stored in cache_idx[k], and is_cached[k] is set to false.
 * In this case AssignCacheIdx should be called, to get an assigned position
 * within the cache set.
 *
 * Cache_time is assigned to the time input argument for all elements in idx.
 *
 * @param [in] keys array of keys that we want to look up in the cache, size [n]
 * @param [in] n number of keys to look up
 * @param [inout] cached_keys keys stored in the cache, size [n_cache_sets * associativity]
 * @param [in] n_cache_sets number of cache sets
 * @param [in] associativity number of keys in cache set
 * @param [inout] cache_time time stamp when the indices were cached, size [n_cache_sets * associativity]
 * @param [out] cache_idx cache indices of the working set elements size [n]
 * @param [out] is_cached  whether the element is cached size[n]
 * @param [in] time iteration counter (used for time stamping)
 */
inline __global__ void get_cache_idx(int *keys, int n, int *cached_keys,
                                     int n_cache_sets, int associativity,
                                     int *cache_time, int *cache_idx,
                                     bool *is_cached, int time) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    int widx = keys[tid];
    int sidx = hash(widx, n_cache_sets, associativity);
    int cidx = sidx * associativity;
    int i = 0;
    bool found = false;
    // search for empty spot and the least recently used spot
    while (i < associativity && !found) {
      found = (cache_time[cidx + i] > 0 && cached_keys[cidx + i] == widx);
      i++;
    }
    is_cached[tid] = found;
    if (found) {
      cidx = cidx + i - 1;
      cache_time[cidx] = time;  //update time stamp
      cache_idx[tid] = cidx;    //exact cache idx
    } else {
      cache_idx[tid] = sidx;  // assign cache set
    }
  }
}

};  // end namespace Cache
};  // end namespace MLCommon
