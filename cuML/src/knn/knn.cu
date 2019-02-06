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

#include "knn.h"
#include <cuda_runtime.h>
#include <iostream>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <vector>



namespace ML {

	kNN::kNN(int D): D(D){}
	kNN::~kNN() {
		for(faiss::gpu::GpuIndexFlatL2* idx : sub_indices) {
			delete idx;
		}

		for(faiss::gpu::StandardGpuResources *r : res) {
			delete r;
		}
	}

	void kNN::fit(float *input, int N, int n_gpus = 1) {

	   for(int dev_no = 0; dev_no < n_gpus; dev_no++) {

			faiss::gpu::GpuIndexFlatConfig config;
			config.device = dev_no;
			config.useFloat16 = false;
			config.storeTransposed = false;

			res.emplace_back(new faiss::gpu::StandardGpuResources());

			sub_indices.emplace_back(
					new faiss::gpu::GpuIndexFlatL2(res[dev_no], D, config)
			);

			indexProxy.addIndex(sub_indices[dev_no]);
	   }

	   indexProxy.add(N, input);
	}

	void kNN::search(float *search_items, int search_items_size, long *res_I, float *res_D, int k) {
		indexProxy.search(search_items_size, search_items, k, res_D, res_I);
	}

/** @} */

}
;
// end namespace ML
