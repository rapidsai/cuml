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

#include "cuda_utils.h"

#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <iostream>


#define CHECK printf("[%d] %s\n", __LINE__, __FILE__);



namespace MLCommon {
namespace Selection {

/** Merge results from several shards into a single result set.
   * @param n number of elements in search array
   * @param k number of neighbors returned
   * @param distances output distance array
   * @param labels output index array
   * @param all_distances  row-wise stacked array of intermediary knn output distances size nshard * n * k
   * @param all_labels     row-wise stacked array of intermediary knn output indices size nshard * n * k
   * @param translations  label translations to apply, size nshard
   */
template <class C>
void merge_tables(long n, long k, long nshard, float *distances, long *labels,
                  float *all_distances, long *all_labels, long *translations) {
  if (k == 0) {
    return;
  }

  size_t stride = n * k;
#pragma omp parallel
  {
    std::vector<int> buf(2 * nshard);
    int *pointer = buf.data();
    ASSERT(pointer != NULL, "Null pointer [%d] %s\n", __LINE__, __FILE__);

    int *shard_ids = pointer + nshard;
    ASSERT(shard_ids != NULL, "Null pointer [%d] %s\n", __LINE__, __FILE__);

    std::vector<float> buf2(nshard);
    float *heap_vals = buf2.data();
    ASSERT(heap_vals != NULL, "Null pointer [%d] %s\n", __LINE__, __FILE__);

#pragma omp for
    for (long i = 0; i < n; i++)
    {
      // the heap maps values to the shard where they are
      // produced.
      const float *D_in = all_distances + i * k;
      const long *I_in = all_labels + i * k;
      int heap_size = 0;

      for (long s = 0; s < nshard; s++) {
        pointer[s] = 0;
        if (I_in[stride * s] >= 0)
          faiss::heap_push<C>(++heap_size, heap_vals, shard_ids,
                              D_in[stride * s], s);
      }

      float *D = distances + i * k;
      long *I = labels + i * k;
      ASSERT(D != NULL and I != NULL, "Null pointer [%d] %s\n", __LINE__, __FILE__);

      for (int j = 0; j < k; j++)
      {
        if (heap_size == 0)
        {
          I[j] = -1;
          D[j] = C::neutral();
        }
        else
        {
          // pop best element
          int s = shard_ids[0];
          int &p = pointer[s];
          D[j] = heap_vals[0];
          I[j] = I_in[stride * s + p] + translations[s];

          faiss::heap_pop<C>(heap_size--, heap_vals, shard_ids);
          p++;
          if (p < k && I_in[stride * s + p] >= 0)
            faiss::heap_push<C>(++heap_size, heap_vals, shard_ids,
                                D_in[stride * s + p], s);
        }
      }

    }
  }
};

/**
   * Search the kNN for the k-nearest neighbors of a set of query vectors
   * @param input device memory to search as an array of device pointers
   * @param sizes array of memory sizes
   * @param n_params size of input and sizes arrays
   * @param D number of cols in input and search_items
   * @param search_items set of vectors to query for neighbors
   * @param n        number of items in search_items
   * @param res_I      pointer to device memory for returning k nearest indices
   * @param res_D      pointer to device memory for returning k nearest distances
   * @param k        number of neighbors to query
   * @param s the cuda stream to use
   */
template <typename IntType = int>
void
brute_force_knn(float **input,
                int *sizes,
                const int n_params,
                const IntType D,
                const float *search_items,
                const IntType n,
                long *res_I,
                float *res_D,
                const IntType k,
                cudaStream_t s)
{
  CHECK;
  std::vector<long> id_ranges;

  IntType total_n = 0;
  for (int i = 0; i < n_params; i++)
  {
    if (i < n_params)  // if i < sizes[i]
      id_ranges.push_back(total_n);
    total_n += sizes[i];
    CHECK;
  }

  CHECK;
  float *result_D = (float*) malloc(sizeof(float) * k * n);
  ASSERT(result_D != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  CHECK;
  long *result_I = (long*) malloc(sizeof(long) * k * n);
  ASSERT(result_I != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  CHECK;
  float *all_D = (float*) malloc(sizeof(float) * n_params * k * n);
  ASSERT(all_D != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  CHECK;
  long *all_I = (long*) malloc(sizeof(long) * n_params * k * n);
  ASSERT(all_I != NULL, "Out of Memory [%d] %s\n", __LINE__, __FILE__);

  CHECK;
  ASSERT_DEVICE_MEM(search_items, "search items");
  ASSERT_DEVICE_MEM(res_I, "output index array");
  ASSERT_DEVICE_MEM(res_D, "output distance array");

  CUDA_CHECK(cudaStreamSynchronize(s));
  CHECK;

  #pragma omp parallel for if(n_params > 1) schedule(static)
  for (int i = 0; i < n_params; i++)
  {
    CHECK;
    const float *ptr = input[i];
    ASSERT(ptr != NULL, "Pointer is NULL [%d] %s\n", __LINE__, __FILE__);

    const IntType size = sizes[i];

    cudaPointerAttributes att;
    cudaError_t err = cudaPointerGetAttributes(&att, ptr);

    CHECK;
    if (err == 0 && att.device > -1)
    {
      CUDA_CHECK(cudaSetDevice(att.device));
      CUDA_CHECK(cudaPeekAtLastError());

      CHECK;
      try
      {
        CHECK;
        faiss::gpu::StandardGpuResources gpu_res;

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CHECK;
        gpu_res.noTempMemory();
        gpu_res.setCudaMallocWarning(false);
        gpu_res.setDefaultStream(att.device, stream);

        faiss::gpu::bruteForceKnn(
          &gpu_res, faiss::METRIC_L2, ptr, true, size, search_items, true,
          n, D, k, all_D + (long(i) * k * long(n)),
          all_I + (long(i) * k * long(n)));

        CHECK;
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaStreamDestroy(stream));
        CHECK;
      }
      catch (const std::exception &e)
      {
        std::cout << "Exception occurred: " << e.what() << std::endl;
      }
    }

    else
    {
      std::stringstream ss;
      ss << "Input memory for " << ptr
         << " failed. isDevice?=" << att.devicePointer << ", N=" << sizes[i];
      std::cout << "Exception: " << ss.str() << std::endl;
    }
  }

  CHECK;
  merge_tables<faiss::CMin<float, IntType>>(
    long(n), k, n_params, result_D, result_I, all_D, all_I, id_ranges.data());

  CHECK;
  MLCommon::updateDevice(res_D, result_D, k * size_t(n), s);

  MLCommon::updateDevice(res_I, result_I, k * size_t(n), s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  CHECK;

  free(all_D);
  free(all_I);

  free(result_D);
  free(result_I);
  CHECK;
};

};  // namespace Selection
};  // namespace MLCommon
