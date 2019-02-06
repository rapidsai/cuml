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

#include "cuda_utils.h"
#include "knn.h"
#include <cuda_runtime.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/Heap.h>

#include <vector>
#include <sstream>


namespace ML {


	/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
	kNN::kNN(int D): D(D), total_n(0), indices(0){}
	kNN::~kNN() {

		for(faiss::gpu::GpuIndexFlatL2* idx : sub_indices) {
			delete idx;
		}

		for(faiss::gpu::GpuResources *r : res) {
			delete r;
		}
	}

	/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
	void kNN::fit(kNNParams *input, int N) {

		for(int i = 0; i < N; i++) {

			kNNParams *params = &input[i];

			cudaPointerAttributes att;
			cudaError_t err = cudaPointerGetAttributes(&att, params->ptr);

			if(err == 0 && att.device > -1) {

				if(i < N)
					id_ranges.push_back(total_n);

				this->total_n += params->N;
				this->indices += 1;

				res.emplace_back(new faiss::gpu::StandardGpuResources());

				faiss::gpu::GpuIndexFlatConfig config;
				config.device = att.device;
				config.useFloat16 = false;
				config.storeTransposed = false;

				sub_indices.emplace_back(
						new faiss::gpu::GpuIndexFlatL2(res[i], D, config)
				);

				// It's only necessary to maintain our set of shards because
				// the GpuIndexFlat class does not support add_with_ids(),
				// a dependency of the IndexShards composite class.
				// As a result, we need to add the ids ourselves
				// and have the reducer/combiner re-label the indices
				// based on the shards they came from.
				sub_indices[i]->add(params->N, params->ptr);
			} else {
				std::stringstream ss;
				ss << "Input memory for " << &params << " failed. isDevice?=" << att.devicePointer;
				throw ss.str();
			}
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

		float *result_D = new float[k*n];
		long *result_I = new long[k*n];

		float *all_D = new float[indices*k*n];
		long *all_I = new long[indices*k*n];

        for(int i = 0; i < indices; i++)
			this->sub_indices[i]->search(n, search_items, k,
					all_D+(i*k*n), all_I+(i*k*n));

		merge_tables<faiss::CMin<float, int>>(n, k, indices,
				result_D, result_I, all_D, all_I, id_ranges.data());

		MLCommon::updateDevice(res_D, result_D, k*n, 0);
		MLCommon::updateDevice(res_I, result_I, k*n, 0);

		delete all_D;
		delete all_I;

		delete result_D;
		delete result_I;
	}


	/** Merge results from several shards into a single result set.
	 * @param all_distances  size nshard * n * k
	 * @param all_labels     idem
	 * @param translartions  label translations to apply, size nshard
	 */
	template <class C>
	void kNN::merge_tables (long n, long k, long nshard,
					   float *distances, long *labels,
					   float *all_distances,
					   long *all_labels,
					   long *translations) {
		if(k == 0) {
			return;
		}

		long stride = n * k;
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
						heap_push<C> (++heap_size, heap_vals, shard_ids,
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

						heap_pop<C> (heap_size--, heap_vals, shard_ids);
						p++;
						if (p < k && I_in[stride * s + p] >= 0)
							heap_push<C> (++heap_size, heap_vals, shard_ids,
										 D_in[stride * s + p], s);
					}
				}
			}
		}

	};

};


// end namespace ML
