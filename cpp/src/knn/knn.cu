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

#include "knn.hpp"

#include "selection/knn.h"

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>


#include <omp.h>
#include <vector>
#include <sstream>


namespace ML {

  void brute_force_knn(
        const cumlHandle &handle,
        float **input, int *sizes, int n_params, int D,
        float *search_items, int n,
        long *res_I, float *res_D, int k) {

    MLCommon::Selection::brute_force_knn(input, sizes, n_params, D,
        search_items, n, res_I, res_D, k, handle.getImpl().getStream());
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
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
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
	 * Search the kNN for the k-nearest neighbors of a set of query vectors
	 * @param search_items set of vectors to query for neighbors
	 * @param n 		   number of items in search_items
	 * @param res_I 	   pointer to device memory for returning k nearest indices
	 * @param res_D		   pointer to device memory for returning k nearest distances
	 * @param k			   number of neighbors to query
	 */
	void kNN::search(float *search_items, int n,
			long *res_I, float *res_D, int k) {

	  MLCommon::Selection::brute_force_knn(ptrs, sizes, indices, D,
	      search_items, n, res_I, res_D, k, handle->getImpl().getStream());
	}

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
    void kNN::fit_from_host(float *ptr, int n, int* devices, int n_chunks) {

        if(this->owner)
          for(int i = 0; i < this->indices; i++) { CUDA_CHECK(cudaFree(this->ptrs[i])); }

        reset();

        this->owner = true;

        float **params = new float*[n_chunks];
        int *sizes = new int[n_chunks];

        MLCommon::chunk_to_device<float>(ptr, n, D, devices, params, sizes, n_chunks, handle->getImpl().getStream());

        fit(params, sizes, n_chunks);
   }
}; // end namespace


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
        }
        catch (...) {
            status = CUML_ERROR_UNKNOWN;
        }
    }
    return status;

}

