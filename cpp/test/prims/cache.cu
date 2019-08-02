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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <iostream>
#include "cache/cache.h"
#include "common/cuml_allocator.hpp"
#include "cuML.hpp"
#include "test_utils.h"

namespace MLCommon {
namespace Cache {

class CacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocator = std::shared_ptr<deviceAllocator>(new defaultDeviceAllocator());
    allocate(x_dev, n_rows * n_cols);
    updateDevice(x_dev, x_host, n_rows * n_cols, stream);
    allocate(tile_dev, n_rows * n_cols);

    allocate(keys_dev, n);
    allocate(is_cached, n);
    allocate(cache_idx_dev, n);
    updateDevice(keys_dev, keys_host, n, stream);
    allocate(zeroone_dev, n);
    allocate(int_array_dev, 12);
    updateDevice(zeroone_dev, zeroone_host, n, stream);
    allocate(argfirst_dev, n_rows);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(x_dev));
    CUDA_CHECK(cudaFree(tile_dev));
    CUDA_CHECK(cudaFree(keys_dev));
    CUDA_CHECK(cudaFree(cache_idx_dev));
    CUDA_CHECK(cudaFree(is_cached));
    CUDA_CHECK(cudaFree(zeroone_dev));
    CUDA_CHECK(cudaFree(int_array_dev));
    CUDA_CHECK(cudaFree(argfirst_dev));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  int n_rows = 10;
  int n_cols = 2;
  int n = 10;

  float *x_dev;
  int *keys_dev;
  int *cache_idx_dev;
  int *int_array_dev;
  float x_host[20] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

  float *tile_dev;

  int keys_host[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  int zeroone_host[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  int *zeroone_dev;

  int *argfirst_dev;

  std::shared_ptr<deviceAllocator> allocator;
  cudaStream_t stream;

  bool *is_cached;
};

__global__ void test_argfirst(const int *array, int n, int *res) {
  int k = threadIdx.x;
  res[k] = arg_first_ge(array, n, k);
}

TEST_F(CacheTest, TestArgFirst) {
  int argfirst_host[10] = {0, 1, 1, 1, 2, 2, 4, 4, 6, 7};
  updateDevice(argfirst_dev, argfirst_host, 10, stream);

  test_argfirst<<<1, 10>>>(argfirst_dev, 10, int_array_dev);
  int idx_exp[10] = {0, 1, 4, 6, 6, 8, 8, 9, 10, 10};
  EXPECT_TRUE(devArrMatchHost(idx_exp, int_array_dev, 10, Compare<int>()));
}

__global__ void test_nth_occurrence(const int *array, int n, int val,
                                    int *res) {
  int k = threadIdx.x;
  res[k] = find_nth_occurrence(array, n, val, k);
}

TEST_F(CacheTest, TestNthOccurrence) {
  test_nth_occurrence<<<1, 10>>>(zeroone_dev, 10, 0, int_array_dev);
  int idx_exp[10] = {0, 1, 2, 3, 4, -1, -1, -1, -1, -1};
  EXPECT_TRUE(devArrMatchHost(idx_exp, int_array_dev, 10, Compare<int>()));
  test_nth_occurrence<<<1, 10>>>(zeroone_dev, 10, 1, int_array_dev);
  int idx_exp2[10] = {5, 6, 7, 8, 9, -1, -1, -1, -1, -1};
  EXPECT_TRUE(devArrMatchHost(idx_exp2, int_array_dev, 10, Compare<int>()));
}

template <int nthreads, int associativity>
__global__ void test_rank_set_entries(const int *array, int n, int *res) {
  const int items_per_thread = ceildiv(associativity, nthreads);
  __shared__ int rank[items_per_thread * nthreads];

  rank_set_entries<nthreads, associativity>(array, n, rank);

  int block_offset = blockIdx.x * associativity;

  for (int i = 0; i < items_per_thread; i++) {
    int k = threadIdx.x * items_per_thread + i;
    if (k < associativity && block_offset + k < n)
      res[block_offset + k] = rank[k];
  }
}

TEST_F(CacheTest, TestRankEntries) {
  // Three cache sets, with 4 elements each
  int val[12] = {12, 11, 10, 9, 8, 6, 7, 5, 4, 1, 2, 3};
  updateDevice(int_array_dev, val, 12, stream);

  const int nthreads = 4;
  test_rank_set_entries<nthreads, 4>
    <<<3, nthreads>>>(int_array_dev, 12, int_array_dev);

  // expect that each block is sorted separately
  // the indices that sorts the block are the following
  int idx_exp[12] = {3, 2, 1, 0, 3, 1, 2, 0, 3, 0, 1, 2};

  EXPECT_TRUE(devArrMatchHost(idx_exp, int_array_dev, 12, Compare<int>()));

  // do the same with less than 4 threads
  const int nthreads3 = 3;
  updateDevice(int_array_dev, val, 12, stream);
  test_rank_set_entries<nthreads3, 4>
    <<<3, nthreads3>>>(int_array_dev, 12, int_array_dev);
  EXPECT_TRUE(devArrMatchHost(idx_exp, int_array_dev, 12, Compare<int>()));
}

TEST_F(CacheTest, TestSimple) {
  float cache_size = 5 * sizeof(float) * n_cols / (1024 * 1024.0);
  Cache<float, 2> cache(allocator, stream, n_cols, cache_size);

  ASSERT_EQ(cache.GetSize(), 4);

  cache.GetCacheIdx(keys_dev, n, cache_idx_dev, is_cached, stream);
  EXPECT_TRUE(devArrMatch(false, is_cached, n, Compare<bool>()));

  int cache_set[10] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  EXPECT_TRUE(devArrMatchHost(cache_set, cache_idx_dev, n, Compare<int>()));
  int n_cached = 1;
  cache.GetCacheIdxPartitioned(keys_dev, n, cache_idx_dev, &n_cached, stream);
  EXPECT_EQ(n_cached, 0);
}

TEST_F(CacheTest, TestAssignCacheIdx) {
  float cache_size = 5 * sizeof(float) * n_cols / (1024 * 1024.0);
  Cache<float, 2> cache(allocator, stream, n_cols, cache_size);

  ASSERT_EQ(cache.GetSize(), 4);

  int n_cached;
  cache.GetCacheIdxPartitioned(keys_dev, n, cache_idx_dev, &n_cached, stream);

  cache.AssignCacheIdx(keys_dev, n, cache_idx_dev, stream);

  int cache_idx_exp[10] = {0, 1, -1, -1, -1, 2, 3, -1, -1, -1};
  int keys_exp[10] = {8, 6, 4, 2, 0, 9, 7, 5, 3, 1};
  EXPECT_TRUE(devArrMatchHost(cache_idx_exp, cache_idx_dev, n, Compare<int>()));
  EXPECT_TRUE(devArrMatchHost(keys_exp, keys_dev, n, Compare<int>()));

  // Now the elements that have been assigned a cache slot are considered cached
  // A subsequent cache lookup should give us their cache indices.
  updateDevice(keys_dev, keys_host, n, stream);
  cache.GetCacheIdxPartitioned(keys_dev, n, cache_idx_dev, &n_cached, stream);
  ASSERT_EQ(n_cached, 4);

  int keys_exp2[4] = {6, 7, 8, 9};
  EXPECT_TRUE(devArrMatchHost(keys_exp2, keys_dev, n_cached, Compare<int>()));
  int cache_idx_exp2[4] = {1, 3, 0, 2};
  EXPECT_TRUE(
    devArrMatchHost(cache_idx_exp2, cache_idx_dev, n_cached, Compare<int>()));

  // Find cache slots, when not available
  int non_cached = n - n_cached;
  cache.AssignCacheIdx(keys_dev + n_cached, non_cached,
                       cache_idx_dev + n_cached, stream);

  int cache_idx_exp3[6] = {-1, -1, -1, -1, -1, -1};
  EXPECT_TRUE(devArrMatchHost(cache_idx_exp3, cache_idx_dev + n_cached,
                              non_cached, Compare<int>()));
}

TEST_F(CacheTest, TestEvict) {
  float cache_size = 8 * sizeof(float) * n_cols / (1024 * 1024.0);
  Cache<float, 4> cache(allocator, stream, n_cols, cache_size);

  ASSERT_EQ(cache.GetSize(), 8);

  int n_cached;
  cache.GetCacheIdxPartitioned(keys_dev, 5, cache_idx_dev, &n_cached, stream);
  ASSERT_EQ(n_cached, 0);
  cache.AssignCacheIdx(keys_dev, 5, cache_idx_dev, stream);

  int cache_idx_exp[5] = {0, 1, 2, 4, 5};
  int keys_exp[5] = {4, 2, 0, 3, 1};
  EXPECT_TRUE(devArrMatchHost(cache_idx_exp, cache_idx_dev, 5, Compare<int>()));
  EXPECT_TRUE(devArrMatchHost(keys_exp, keys_dev, 5, Compare<int>()));

  int idx_host[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  updateDevice(keys_dev, idx_host, 10, stream);
  cache.GetCacheIdxPartitioned(keys_dev, 10, cache_idx_dev, &n_cached, stream);
  EXPECT_EQ(n_cached, 3);
  int cache_idx_exp2[3] = {1, 4, 0};
  EXPECT_TRUE(
    devArrMatchHost(cache_idx_exp2, cache_idx_dev, 3, Compare<int>()));

  cache.AssignCacheIdx(keys_dev + n_cached, 10 - n_cached,
                       cache_idx_dev + n_cached, stream);

  int keys_exp3[10] = {2, 3, 4, 10, 8, 6, 11, 9, 7, 5};
  int cache_idx_exp3[10] = {1, 4, 0, 3, 2, -1, 6, 7, 5, -1};
  EXPECT_TRUE(devArrMatchHost(keys_exp3, keys_dev, 10, Compare<int>()));
  EXPECT_TRUE(
    devArrMatchHost(cache_idx_exp3, cache_idx_dev, 10, Compare<int>()));
}

TEST_F(CacheTest, TestStoreCollect) {
  float cache_size = 8 * sizeof(float) * n_cols / (1024 * 1024.0);
  Cache<float, 4> cache(allocator, stream, n_cols, cache_size);

  ASSERT_EQ(cache.GetSize(), 8);

  int n_cached;

  cache.GetCacheIdxPartitioned(keys_dev, 5, cache_idx_dev, &n_cached, stream);
  cache.AssignCacheIdx(keys_dev, 5, cache_idx_dev, stream);
  cache.GetCacheIdxPartitioned(keys_dev, 5, cache_idx_dev, &n_cached, stream);

  cache.StoreVecs(x_dev, 10, n_cached, cache_idx_dev, stream, keys_dev);
  cache.GetCacheIdxPartitioned(keys_dev, 5, cache_idx_dev, &n_cached, stream);
  cache.GetVecs(cache_idx_dev, n_cached, tile_dev, stream);

  int cache_idx_host[10];
  updateHost(cache_idx_host, cache_idx_dev, n_cached, stream);
  int keys_host[10];
  updateHost(keys_host, keys_dev, n_cached, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < n_cached; i++) {
    EXPECT_TRUE(devArrMatch(x_dev + keys_host[i] * n_cols,
                            tile_dev + i * n_cols, n_cols, Compare<int>()))
      << "vector " << i;
  }

  for (int k = 0; k < 4; k++) {
    cache.GetCacheIdxPartitioned(keys_dev, 10, cache_idx_dev, &n_cached,
                                 stream);
    if (k == 0) {
      EXPECT_EQ(n_cached, 5);
    } else {
      EXPECT_EQ(n_cached, 8);
    }

    cache.AssignCacheIdx(keys_dev + n_cached, 10 - n_cached,
                         cache_idx_dev + n_cached, stream);
    cache.StoreVecs(x_dev, 10, 10 - n_cached, cache_idx_dev + n_cached, stream,
                    keys_dev + n_cached);

    cache.GetVecs(cache_idx_dev, 10, tile_dev, stream);

    updateHost(cache_idx_host, cache_idx_dev, 10, stream);
    updateHost(keys_host, keys_dev, 10, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < 10; i++) {
      if (cache_idx_host[i] >= 0) {
        EXPECT_TRUE(devArrMatch(x_dev + keys_host[i] * n_cols,
                                tile_dev + i * n_cols, n_cols, Compare<int>()))
          << "vector " << i;
      }
    }
  }
}
};  // end namespace Cache
};  // end namespace MLCommon
