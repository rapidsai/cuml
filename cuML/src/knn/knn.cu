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
#include <faiss/gpu/GpuDistance.h>

#include <omp.h>
#include <vector>
#include <sstream>


namespace ML {


	/**
	 * Build a kNN object for training and querying a k-nearest neighbors model.
	 * @param D 	number of features in each vector
	 */
	kNN::kNN(int D, bool verbose): D(D), total_n(0), indices(0), verbose(verbose), owner(false){}
	kNN::~kNN() {

	    if(this->owner) {
            for(kNNParams p : knn_params) { CUDA_CHECK(cudaFree(p.ptr)); }
	    }
	}

	void kNN::reset() {

        if(knn_params.size() > 0) {
            knn_params.clear();
            this->indices = 0;
            this->total_n = 0;
        }
	}

	bool kNN::verify_size(size_t size, int device) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        if(size > free) {
            std::cout << "Not enough free memory on device "
                    << device
                    << " to run kneighbors. "
                    << "needed="
                    << size
                    << ", free=" << free << std::endl;
            return false;
        }

        return true;
	}

	/**
	 * Fit a kNN model by creating separate indices for multiple given
	 * instances of kNNParams.
	 * @param input  an array of pointers to data on (possibly different) devices
	 * @param N 	 number of items in input array.
	 */
	void kNN::fit(kNNParams *input, int N) {

        if(this->owner) {
            for(kNNParams p : knn_params) { CUDA_CHECK(cudaFree(p.ptr)); }
        }

	    if(this->verbose)
	        std::cout << "N=" << N << std::endl;

	    reset();

        for(int i = 0; i < N; i++) {

            kNNParams params = input[i];
            this->indices++;
            this->knn_params.emplace_back(params);
            if(i < params.N) {
                id_ranges.push_back(total_n);
            }

            this->total_n += params.N;
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
	void kNN::search(const float *search_items, int n,
			long *res_I, float *res_D, int k) {

		float *result_D = new float[k*size_t(n)];
		long*result_I = new long[k*size_t(n)];

		float *all_D = new float[indices*k*size_t(n)];
		long *all_I = new long[indices*k*size_t(n)];


		/**
		 * Initial verification of memory
		 */
		for(int i = 0; i < indices; i++) {
            kNNParams params = knn_params[i];

            cudaPointerAttributes att;
            cudaError_t err = cudaPointerGetAttributes(&att, params.ptr);

            if(err == 0 && att.device > -1) {
                CUDA_CHECK(cudaSetDevice(att.device));

                if(!verify_size(size_t(params.N)*size_t(this->D)*4l, att.device))
                    return;
            }
		}


        #pragma omp parallel
		{
            #pragma omp for
            for(int i = 0; i < indices; i++) {

                kNNParams params = knn_params[i];

                cudaPointerAttributes att;
                cudaError_t err = cudaPointerGetAttributes(&att, params.ptr);

                if(err == 0 && att.device > -1) {
                    CUDA_CHECK(cudaSetDevice(att.device));

                    try {
                        faiss::gpu::StandardGpuResources gpu_res;
                        gpu_res.setTempMemory(size_t(params.N)*size_t(this->D)*4l);

                        bruteForceKnn(&gpu_res,
                                    faiss::METRIC_L2,
                                    params.ptr,
                                    params.N,
                                    search_items,
                                    n,
                                    this->D,
                                    k,
                                    all_D+(long(i)*k*long(n)),
                                    all_I+(long(i)*k*long(n)));

                        CUDA_CHECK(cudaPeekAtLastError());

                    } catch(const std::exception &e) {
                       std::cout << "Exception occurred: " << e.what() << std::endl;
                    }


                } else {
                    std::stringstream ss;
                    ss << "Input memory for " << &params << " failed. isDevice?=" << att.devicePointer << ", N=" << params.N;
                    std::cout << "Exception: " << ss.str() << std::endl;
                }
            }
		}


		merge_tables<faiss::CMin<float, int>>(long(n), k, indices,
				result_D, result_I, all_D, all_I, id_ranges.data());

		MLCommon::updateDevice(res_D, result_D, k*size_t(n), 0);
		MLCommon::updateDevice(res_I, result_I, k*size_t(n), 0);

		std::cout << MLCommon::arr2Str(res_D, k*size_t(n), "res_D") << std::endl;
        std::cout << MLCommon::arr2Str(res_I, k*size_t(n), "res_I") << std::endl;

		delete all_D;
		delete all_I;

		delete result_D;
		delete result_I;
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

        if(this->owner) {
            for(kNNParams p : knn_params) { CUDA_CHECK(cudaFree(p.ptr)); }
        }

        reset();

        size_t chunk_size = MLCommon::ceildiv<size_t>((size_t)n, (size_t)n_chunks);
        kNNParams params[n_chunks];

        this->owner = true;

        /**
         * Initial verification of memory
         */
        for(int i = 0; i < n_chunks; i++) {

            int device = devices[i];
            size_t length = chunk_size;
            if(length * i >= n)
                length = (chunk_size*i)-size_t(n);
            CUDA_CHECK(cudaSetDevice(device));
            if(!verify_size(size_t(length)*size_t(D), device))
                return;
        }



        #pragma omp parallel for
        for(int i = 0; i < n_chunks; i++) {

            int device = devices[i];
            CUDA_CHECK(cudaSetDevice(device));

            size_t length = chunk_size;
            if(length * i >= n)
                length = (size_t(chunk_size)*i)-size_t(n);

            float *ptr_d;
            MLCommon::allocate(ptr_d, size_t(length)*size_t(D));
            MLCommon::updateDevice(ptr_d, ptr+(size_t(chunk_size)*i), size_t(length)*size_t(D));

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
	void kNN::merge_tables (long n, long k, long nshard,
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
