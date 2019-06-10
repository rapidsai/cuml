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

#include "common/cumlHandle.hpp"

<<<<<<< HEAD
#include "knn.hpp"

#include "knn/util.h"

#include "selection/knn.h"

#include "cuda_utils.h"
=======
>>>>>>> branch-0.8
#include <cuda_runtime.h>
#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include "cuda_utils.h"
#include "knn.h"

<<<<<<< HEAD
#include <vector>
=======
#include <omp.h>
>>>>>>> branch-0.8
#include <sstream>
#include <vector>

namespace ML {

<<<<<<< HEAD
  /**
   * @brief Flat C++ API function to perform a brute force knn on
   * a series of input arrays and combine the results into a single
   * output array for indexes and distances.
   *
   * @param handle the cuml handle to use
   * @param input an array of pointers to the input arrays
   * @param sizes an array of sizes of input arrays
   * @param n_params array size of input and sizes
   * @param D the dimensionality of the arrays
   * @param search_items array of items to search of dimensionality D
   * @param n number of rows in search_items
   * @param res_I the resulting index array of size n * k
   * @param res_D the resulting distance array of size n * k
   * @param k the number of nearest neighbors to return
   */
  void brute_force_knn(
        cumlHandle &handle,
        float **input, int *sizes, int n_params, int D,
        float *search_items, int n,
        long *res_I, float *res_D, int k) {
    MLCommon::Selection::brute_force_knn(input, sizes, n_params, D,
        search_items, n, res_I, res_D, k, handle.getImpl().getStream());
  }


  /**
   * @brief A flat C++ API function that chunks a host array up into
   * some number of different devices
   *
   * @param ptr an array on host to chunk
   * @param n number of rows in host array
   * @param D number of cols in host array
   * @param devices array of devices to use
   * @param output an array of output pointers to allocate and use
   * @param sizes output array sizes
   * @param n_chunks number of chunks to spread across device arrays
   */
  void chunk_host_array(
    cumlHandle &handle,
    const float *ptr, int n, int D,
    int* devices, float **output, int *sizes, int n_chunks) {
    chunk_to_device<float, int>(ptr, n, D,
        devices, output, sizes, n_chunks, handle.getImpl().getStream());
  }

	/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
	kNN::kNN(const cumlHandle &handle, int D, bool verbose):
	        D(D), total_n(0), indices(0), verbose(verbose), owner(false) {
	    this->handle = const_cast<cumlHandle*>(&handle);
	    sizes = nullptr;
	    ptrs = nullptr;
	}

	kNN::~kNN() {

	    try {
	        if(this->owner) {
	            if(this->verbose)
	                std::cout << "Freeing kNN memory" << std::endl;
	            for(int i = 0; i < this->indices; i++) { CUDA_CHECK(cudaFree(this->ptrs[i])); }
	        }

	    } catch(const std::exception &e) {
	        std::cout << "An exception occurred releasing kNN memory: " << e.what() << std::endl;
	    }

	    delete ptrs;
	    delete sizes;
	}

	void kNN::reset() {
        if(this->indices > 0) {
            this->indices = 0;
            this->total_n = 0;
        }
	}

	/**
=======
/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
kNN::kNN(const cumlHandle &handle, int D, bool verbose)
  : D(D), total_n(0), indices(0), verbose(verbose), owner(false) {
  this->handle = const_cast<cumlHandle_impl *>(&handle.getImpl());
}

kNN::~kNN() {
  try {
    if (this->owner) {
      if (this->verbose) std::cout << "Freeing kNN memory" << std::endl;
      for (kNNParams p : knn_params) {
        CUDA_CHECK(cudaFree(p.ptr));
      }
    }

  } catch (const std::exception &e) {
    std::cout << "An exception occurred releasing kNN memory: " << e.what()
              << std::endl;
  }
}

void kNN::reset() {
  if (knn_params.size() > 0) {
    knn_params.clear();
    this->id_ranges.clear();
    this->indices = 0;
    this->total_n = 0;
  }
}

bool kNN::verify_size(size_t size, int device) {
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  if (size > free) {
    std::cout << "Not enough free memory on device " << device
              << " to run kneighbors. "
              << "needed=" << size << ", free=" << free << std::endl;
    return false;
  }

  return true;
}

/**
>>>>>>> branch-0.8
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
<<<<<<< HEAD
	void kNN::fit(float **input, int *sizes, int N) {

      if(this->owner)
        for(int i = 0; i < this->indices; i++) { CUDA_CHECK(cudaFree(this->ptrs[i])); }

	    if(this->verbose)
	        std::cout << "N=" << N << std::endl;

	    reset();

	    // TODO: Copy pointers!
	    this->indices = N;
	    this->ptrs = (float**)malloc(N*sizeof(float*));
	    this->sizes = (int*)malloc(N*sizeof(int));

	    for(int i = 0; i < N; i++) {
	      this->ptrs[i] = input[i];
	      this->sizes[i] = sizes[i];
	    }
	}

	/**
=======
void kNN::fit(kNNParams *input, int N) {
  if (this->owner)
    for (kNNParams p : knn_params) {
      CUDA_CHECK(cudaFree(p.ptr));
    }

  if (this->verbose) std::cout << "N=" << N << std::endl;

  reset();

  for (int i = 0; i < N; i++) {
    kNNParams params = input[i];
    this->indices++;
    this->knn_params.emplace_back(params);
    if (i < params.N) {
      id_ranges.push_back(total_n);
    }

    this->total_n += params.N;
  }
}

template <typename T>
void ASSERT_MEM(T *ptr, std::string name) {
  cudaPointerAttributes s_att;
  cudaError_t s_err = cudaPointerGetAttributes(&s_att, ptr);

  if (s_err != 0 || s_att.device == -1)
    std::cout << "Invalid device pointer encountered in " << name
              << ". device=" << s_att.device << ", err=" << s_err << std::endl;
}

/**
>>>>>>> branch-0.8
	 * Search the kNN for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 */
<<<<<<< HEAD
	void kNN::search(float *search_items, int n,
			long *res_I, float *res_D, int k) {

	  MLCommon::Selection::brute_force_knn(ptrs, sizes, indices, D,
	      search_items, n, res_I, res_D, k, handle->getImpl().getStream());
	}
=======
void kNN::search(const float *search_items, int n, long *res_I, float *res_D,
                 int k) {
  float *result_D = new float[k * size_t(n)];
  long *result_I = new long[k * size_t(n)];

  float *all_D = new float[indices * k * size_t(n)];
  long *all_I = new long[indices * k * size_t(n)];

  ASSERT_MEM(search_items, "search items");
  ASSERT_MEM(res_I, "output index array");
  ASSERT_MEM(res_D, "output distance array");

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < indices; i++) {
      kNNParams params = knn_params[i];

      cudaPointerAttributes att;
      cudaError_t err = cudaPointerGetAttributes(&att, params.ptr);

      if (err == 0 && att.device > -1) {
        CUDA_CHECK(cudaSetDevice(att.device));
        CUDA_CHECK(cudaPeekAtLastError());

        try {
          faiss::gpu::StandardGpuResources gpu_res;

          cudaStream_t stream;
          CUDA_CHECK(cudaStreamCreate(&stream));

          gpu_res.noTempMemory();
          gpu_res.setCudaMallocWarning(false);
          gpu_res.setDefaultStream(att.device, stream);

          bruteForceKnn(&gpu_res, faiss::METRIC_L2, params.ptr, params.N,
                        search_items, n, this->D, k,
                        all_D + (long(i) * k * long(n)),
                        all_I + (long(i) * k * long(n)));

          CUDA_CHECK(cudaPeekAtLastError());
          CUDA_CHECK(cudaStreamSynchronize(stream));

          CUDA_CHECK(cudaStreamDestroy(stream));

        } catch (const std::exception &e) {
          std::cout << "Exception occurred: " << e.what() << std::endl;
        }

      } else {
        std::stringstream ss;
        ss << "Input memory for " << &params
           << " failed. isDevice?=" << att.devicePointer << ", N=" << params.N;
        std::cout << "Exception: " << ss.str() << std::endl;
      }
    }
  }

  merge_tables<faiss::CMin<float, int>>(long(n), k, indices, result_D, result_I,
                                        all_D, all_I, id_ranges.data());

  MLCommon::updateDevice(res_D, result_D, k * size_t(n), handle->getStream());
  MLCommon::updateDevice(res_I, result_I, k * size_t(n), handle->getStream());

  delete all_D;
  delete all_I;

  delete result_D;
  delete result_I;
}
>>>>>>> branch-0.8

/**
     * Chunk a host array up into one or many GPUs (determined by the provided
     * list of gpu ids) and fit a knn model.
     *
     * @param ptr       an array in host memory to chunk over devices
     * @param n         number of elements in ptr
     * @param devices   array of device ids for chunking the ptr
     * @param n_chunks  number of elements in gpus
     * @param out       host pointer (size n) to store output
     */
<<<<<<< HEAD
    void kNN::fit_from_host(float *ptr, int n, int* devices, int n_chunks) {

        if(this->owner)
          for(int i = 0; i < this->indices; i++) { CUDA_CHECK(cudaFree(this->ptrs[i])); }
=======
void kNN::fit_from_host(float *ptr, int n, int *devices, int n_chunks) {
  if (this->owner) {
    for (kNNParams p : knn_params) {
      CUDA_CHECK(cudaFree(p.ptr));
    }
  }
>>>>>>> branch-0.8

  reset();

<<<<<<< HEAD
        this->owner = true;

        float **params = new float*[n_chunks];
        int *sizes = new int[n_chunks];

        chunk_to_device<float>(ptr, n, D, devices, params, sizes, n_chunks, handle->getImpl().getStream());

        fit(params, sizes, n_chunks);
   }
}; // end namespace


/**
 * @brief Flat C API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param handle the cuml handle to use
 * @param input an array of pointers to the input arrays
 * @param sizes an array of sizes of input arrays
 * @param n_params array size of input and sizes
 * @param D the dimensionality of the arrays
 * @param search_items array of items to search of dimensionality D
 * @param n number of rows in search_items
 * @param res_I the resulting index array of size n * k
 * @param res_D the resulting distance array of size n * k
 * @param k the number of nearest neighbors to return
 */
extern "C" cumlError_t knn_search(
    const cumlHandle_t handle,
    float **input, int *sizes, int n_params, int D,
    float *search_items, int n,
    long *res_I, float *res_D, int k) {

    cumlError_t status;

    ML::cumlHandle *handle_ptr;
    std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
    if (status == CUML_SUCCESS) {
        try {
            MLCommon::Selection::brute_force_knn(input, sizes, n_params, D,
                search_items, n,
                res_I, res_D, k,
                handle_ptr->getImpl().getStream());
        } catch (...) {
            status = CUML_ERROR_UNKNOWN;
        }
    }
    return status;
}

/**
 * @brief A flat C api function that chunks a host array up into
 * some number of different devices
 *
 * @param ptr an array on host to chunk
 * @param n number of rows in host array
 * @param D number of cols in host array
 * @param devices array of devices to use
 * @param output an array of output pointers to allocate and use
 * @param sizes output array sizes
 * @param n_chunks number of chunks to spread across device arrays
 */
extern "C" cumlError_t chunk_host_array(
    const cumlHandle_t handle,
    const float *ptr, int n, int D,
    int* devices, float **output, int *sizes, int n_chunks) {

  cumlError_t status;
  ML::cumlHandle *handle_ptr;
  std::tie(handle_ptr, status) = ML::handleMap.lookupHandlePointer(handle);
  if(status == CUML_SUCCESS) {
    try {
      ML::chunk_to_device<float, int>(ptr, n, D,
          devices, output, sizes, n_chunks, handle_ptr->getImpl().getStream());
    } catch (...) {
        status = CUML_ERROR_UNKNOWN;
    }
  }
  return status;
}



=======
  size_t chunk_size = MLCommon::ceildiv<size_t>((size_t)n, (size_t)n_chunks);
  kNNParams params[n_chunks];

  this->owner = true;

  /**
         * Initial verification of memory
         */
  for (int i = 0; i < n_chunks; i++) {
    int device = devices[i];
    size_t length = chunk_size;
    if (length * i >= n) length = (chunk_size * i) - size_t(n);
    CUDA_CHECK(cudaSetDevice(device));
    if (!verify_size(size_t(length) * size_t(D), device)) return;
  }

#pragma omp parallel for
  for (int i = 0; i < n_chunks; i++) {
    int device = devices[i];
    CUDA_CHECK(cudaSetDevice(device));

    size_t length = chunk_size;
    if (length * i >= n) length = (size_t(chunk_size) * i) - size_t(n);

    float *ptr_d;
    MLCommon::allocate(ptr_d, size_t(length) * size_t(D));
    MLCommon::updateDevice(ptr_d, ptr + (size_t(chunk_size) * i),
                           size_t(length) * size_t(D), handle->getStream());

    kNNParams p;
    p.N = length;
    p.ptr = ptr_d;

    params[i] = p;
  }

  fit(params, n_chunks);
}

/** Merge results from several shards into a single result set.
	 * @param all_distances  size nshard * n * k
	 * @param all_labels     idem
	 * @param translartions  label translations to apply, size nshard
	 */
template <class C>
void kNN::merge_tables(long n, long k, long nshard, float *distances,
                       long *labels, float *all_distances, long *all_labels,
                       long *translations) {
  if (k == 0) {
    return;
  }

  size_t stride = n * k;
#pragma omp parallel
  {
    std::vector<int> buf(2 * nshard);
    int *pointer = buf.data();
    int *shard_ids = pointer + nshard;
    std::vector<float> buf2(nshard);
    float *heap_vals = buf2.data();
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
          heap_push<C>(++heap_size, heap_vals, shard_ids, D_in[stride * s], s);
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
          int &p = pointer[s];
          D[j] = heap_vals[0];
          I[j] = I_in[stride * s + p] + translations[s];

          heap_pop<C>(heap_size--, heap_vals, shard_ids);
          p++;
          if (p < k && I_in[stride * s + p] >= 0)
            heap_push<C>(++heap_size, heap_vals, shard_ids,
                         D_in[stride * s + p], s);
        }
      }
    }
  }
};

};  // namespace ML

// end namespace ML
>>>>>>> branch-0.8
