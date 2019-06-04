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

#include "common/cumlHandle.hpp"

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <faiss/gpu/GpuIndexFlat.h>

#include <faiss/gpu/GpuResources.h>
#include <faiss/Heap.h>
#include <faiss/gpu/GpuDistance.h>

#include <iostream>

namespace ML {


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
         float **input, int*sizes, int n_params, int D,
         float *search_items, int n,
         long *res_I, float *res_D, int k);

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
    int* devices, float **output, int *sizes, int n_chunks);


	class kNN {

		float **ptrs;
		int *sizes;

		int total_n;
		int indices;
		int D;
		bool verbose;
		bool owner;

		cumlHandle *handle;

    public:
	    /**
	     * Build a kNN object for training and querying a k-nearest neighbors model.
	     * @param D     number of features in each vector
	     */
		kNN(const cumlHandle &handle, int D, bool verbose = false);
		~kNN();

    void reset();


    /**
     * Search the kNN for the k-nearest neighbors of a set of query vectors
     * @param search_items set of vectors to query for neighbors
     * @param n            number of items in search_items
     * @param res_I        pointer to device memory for returning k nearest indices
     * @param res_D        pointer to device memory for returning k nearest distances
     * @param k            number of neighbors to query
     */
		void search(float *search_items, int search_items_size,
		        long *res_I, float *res_D, int k);

    /**
     * Fit a kNN model by creating separate indices for multiple given
     * instances of kNNParams.
     * @param input  an array of pointers to data on (possibly different) devices
     * @param N      number of items in input array.
     */
		void fit(float **input, int *sizes, int N);

		/**
		 * Chunk a host array up into one or many GPUs (determined by the provided
		 * list of gpu ids) and fit a knn model.
		 *
		 * @param ptr       an array in host memory to chunk over devices
		 * @param n         number of elements in ptr
		 * @param gpus      array of device ids for chunking the ptr
		 * @param n_chunks  number of elements in gpus
		 * @param out       host pointer to copy output
		 */
		void fit_from_host(float *ptr, int n, int* devices, int n_chunks);
  };
};

