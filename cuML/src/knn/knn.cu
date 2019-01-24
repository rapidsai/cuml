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


namespace ML {

	using namespace faiss;
	using namespace faiss::gpu;

	kNN::kNN(int D): D(D), total_n(0), indices(0){}
	kNN::~kNN() {

		for(GpuIndexFlatL2* idx : sub_indices) {
			delete idx;
		}

		for(GpuResources *r : res) {
			delete r;
		}
	}

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

				auto idx = new faiss::gpu::GpuIndexFlatL2(res[i], D, config);
				idx->add(params->N, params->ptr);

				sub_indices.emplace_back(idx);
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

        for(int i = 0; i < indices; i++)
			this->sub_indices[i]->search(n, search_items, k, all_D+(i*k*n), all_I+(i*k*n));

		merge_tables<CMin<float, int>>(n, k, indices, result_D, result_I, all_D, all_I, id_ranges.data());

		updateDevice(res_D, result_D, k*n, 0);
		updateDevice(res_I, result_I, k*n, 0);

		delete all_D;
		delete all_I;

		delete result_D;
		delete result_I;
	}
}
;


// end namespace ML
