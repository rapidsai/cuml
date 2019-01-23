/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "knn_c.h"
#include <cuda_runtime.h>
#include <iostream>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/FaissException.h>
#include <faiss/MetaIndexes.h>
#include <faiss/Heap.h>
#include <vector>
#include <cuda_utils.h>



namespace ML {

	using namespace faiss;
	using namespace faiss::gpu;

	kNN::kNN(int D): D(D), total_n(0), indices(0), indexShards(D, true, false){}
	kNN::~kNN() {

		std::cout << "Destructor called" << std::endl;
		for(GpuIndexFlatL2* idx : sub_indices) {
			delete idx;
		}

		for(GpuResources *r : res) {
			delete r;
		}
	}

	/**
	 *
	 */
	void kNN::fit(kNNParams *input, int N) {

		// Loop through different inputs

		for(int i = 0; i < N; i++) {

			kNNParams params = input[i];

			this->total_n += params.N;
			this->indices += 1;

			cudaPointerAttributes att;
			cudaError_t err = cudaPointerGetAttributes(&att, params.ptr);

			if(err == 0 && att.device > -1) {

				res.emplace_back(new faiss::gpu::StandardGpuResources());

				faiss::gpu::GpuIndexFlatConfig config;
				config.device = att.device;
				config.useFloat16 = false;
				config.storeTransposed = false;

				auto idx = new faiss::gpu::GpuIndexFlatL2(res[i], D, config);
				idx->verbose = true;

				// initialize ids
				long *ids = new long[params.N];
				for(int j = 0; j < params.N; j++)
					ids[j] = j*(i+1);

				sub_indices.emplace_back(idx);
				idx->add(params.N, params.ptr);
				indexShards.add_shard(sub_indices[i]);
			} else {
				// Throw error- we don't have device memory
			}
		}
	}

	void kNN::search(float *search_items, int n, long *res_I, float *res_D, int k) {

		float *result_D = new float[k*n];
		long *result_I = new long[k*n];

		float *all_D = new float[indices*k*n];
		long *all_I = new long[indices*k*n];

	    std::vector<long> translations (indices, 0);
        translations[0] = 0;

        for(int i = 0; i < indices; i++) {
			this->sub_indices[i]->search(n, search_items, k, all_D+(i*k*n), all_I+(i*k*n));

			std::cout << all_D+(i*k*n) << std::endl;

			if(i+1 < indices) {
				translations [i + 1] = translations [i] +
					sub_indices [i]->ntotal;
			}
		}

		merge_tables<CMin<float, int>>(n, k, indices, result_D, result_I, all_D, all_I, translations.data());

		MLCommon::updateDevice(res_D, result_D, k*n, 0);
		MLCommon::updateDevice(res_I, result_I, k*n, 0);
	}
}
;


// end namespace ML
