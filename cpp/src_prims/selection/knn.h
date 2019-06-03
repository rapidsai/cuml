#pragma once

#include "cuda_utils.h"

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/Heap.h>

#include <iostream>


namespace MLCommon {
namespace Selection {



  /** Merge results from several shards into a single result set.
   * @param all_distances  size nshard * n * k
   * @param all_labels     idem
   * @param translartions  label translations to apply, size nshard
   */
  template <class C>
  void merge_tables (long n, long k, long nshard,
             float *distances, long *labels,
             float *all_distances,
             long *all_labels,
             long *translations) {
    if(k == 0) {
      return;
    }

    size_t stride = n * k;
    #pragma omp parallel
    {
      std::vector<int> buf (2 * nshard);
      int * pointer = buf.data();
      int * shard_ids = pointer + nshard;
      std::vector<float> buf2 (nshard);
      float * heap_vals = buf2.data();
      #pragma omp for
      for (long i = 0; i < n; i++) {
        // the heap maps values to the shard where they are
        // produced.
        const float *D_in = all_distances + i * k;
        const long *I_in = all_labels + i * k;
        int heap_size = 0;

        for (long s = 0; s < nshard; s++) {
          pointer[s] = 0;
          if (I_in[stride * s] >= 0)
            faiss::heap_push<C> (++heap_size, heap_vals, shard_ids,
                   D_in[stride * s], s);
        }

        float *D = distances + i * k;
        long *I = labels + i * k;

        for (int j = 0; j < k; j++) {
          if (heap_size == 0) {
            I[j] = -1;
            D[j] = C::neutral();
          } else {
            // pop best element
            int s = shard_ids[0];
            int & p = pointer[s];
            D[j] = heap_vals[0];
            I[j] = I_in[stride * s + p] + translations[s];

            faiss::heap_pop<C> (heap_size--, heap_vals, shard_ids);
            p++;
            if (p < k && I_in[stride * s + p] >= 0)
              faiss::heap_push<C> (++heap_size, heap_vals, shard_ids,
                     D_in[stride * s + p], s);
          }
        }
      }
    }
  };

  /**
   * Search the kNN for the k-nearest neighbors of a set of query vectors
   * @param search_items set of vectors to query for neighbors
   * @param n        number of items in search_items
   * @param res_I      pointer to device memory for returning k nearest indices
   * @param res_D      pointer to device memory for returning k nearest distances
   * @param k        number of neighbors to query
   */
  template<typename IntType = int>
  void brute_force_knn(
      float **input, int *sizes, int n_params, IntType D,
      float *search_items, IntType n,
      long *res_I, float *res_D, IntType k,
      cudaStream_t s) {

    std::vector<long> *id_ranges = new std::vector<long>();

    IntType total_n = 0;

    for(int i = 0; i < n_params; i++) {
      if(i < n_params) // if i < sizes[i]
          id_ranges->push_back(total_n);
      total_n += sizes[i];
    }

    float *result_D = new float[k*size_t(n)];
    long*result_I = new long[k*size_t(n)];

    float *all_D = new float[n_params*k*size_t(n)];
    long *all_I = new long[n_params*k*size_t(n)];

    ASSERT_MEM(search_items, "search items");
    ASSERT_MEM(res_I, "output index array");
    ASSERT_MEM(res_D, "output distance array");


    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < n_params; i++) {

          const float *ptr = input[i];
          IntType size = sizes[i];

          cudaPointerAttributes att;
          cudaError_t err = cudaPointerGetAttributes(&att, ptr);

          if(err == 0 && att.device > -1) {

              CUDA_CHECK(cudaSetDevice(att.device));
              CUDA_CHECK(cudaPeekAtLastError());

              try {

                  faiss::gpu::StandardGpuResources gpu_res;

                  cudaStream_t stream;
                  CUDA_CHECK(cudaStreamCreate(&stream));

                  gpu_res.noTempMemory();
                  gpu_res.setCudaMallocWarning(false);
                  gpu_res.setDefaultStream(att.device, stream);

                  faiss::gpu::bruteForceKnn(&gpu_res,
                              faiss::METRIC_L2,
                              ptr,
                              size,
                              search_items,
                              n,
                              D,
                              k,
                              all_D+(long(i)*k*long(n)),
                              all_I+(long(i)*k*long(n)));

                  CUDA_CHECK(cudaPeekAtLastError());
                  CUDA_CHECK(cudaStreamSynchronize(stream));

                  CUDA_CHECK(cudaStreamDestroy(stream));

              } catch(const std::exception &e) {
                 std::cout << "Exception occurred: " << e.what() << std::endl;
              }

          } else {
              std::stringstream ss;
              ss << "Input memory for " << ptr << " failed. isDevice?=" <<
                  att.devicePointer << ", N=" << sizes[i];
              std::cout << "Exception: " << ss.str() << std::endl;
          }
      }
    }

    merge_tables<faiss::CMin<float, IntType>>(long(n), k, n_params,
        result_D, result_I, all_D, all_I, id_ranges->data());

    MLCommon::updateDevice(res_D, result_D, k*size_t(n), s);
    MLCommon::updateDevice(res_I, result_I, k*size_t(n), s);

    delete all_D;
    delete all_I;

    delete result_D;
    delete result_I;
  };

}; // end namespace SELECTION
}; // end namespace MLCOMMON
