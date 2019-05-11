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

#include "cuML.hpp"
#include <common/cumlHandle.hpp>
#include <common/host_buffer.hpp>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <iostream>
#include <faiss/gpu/GpuIndexFlat.h>

#pragma once

namespace ML {

	using namespace faiss;

	class kNNParams {
	public:
		float *ptr;
		size_t N;
		friend std::ostream & operator<<(std::ostream &str, kNNParams &v) {
			str << "kNNParams {ptr=" << v.ptr << ", N=" << v.N << "}";
			return str;
		}
	};


    class kNN {

		std::vector<long> id_ranges;

		std::vector<kNNParams> knn_params;

		size_t total_n;
		int indices;
		int D;
		bool verbose;
		bool owner;

		cumlHandle_impl handle;



    public:
	    /**
	     * Build a kNN object for training and querying a k-nearest neighbors model.
	     * @param D     number of features in each vector
	     */
		kNN(const cumlHandle &handle,  int D, bool verbose = false);
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
		void search(const float *search_items, size_t search_items_size,
		        long *res_I, float *res_D, size_t k);

        /**
         * Fit a kNN model by creating separate indices for multiple given
         * instances of kNNParams.
         * @param input  an array of pointers to data on (possibly different) devices
         * @param N      number of items in input array.
         */
		void fit(kNNParams *input, size_t N);

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
		void fit_from_host(float *ptr, size_t n, int* devices, size_t n_chunks);

		template <class C>
		void merge_tables(long n, long k, long nshard,
							   float *distances, long *labels,
							   float *all_distances,
							   long *all_labels,
							   long *translations);
    };
}

