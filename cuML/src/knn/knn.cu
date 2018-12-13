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
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

namespace ML {

using namespace MLCommon;

class kNN {

    faiss::gpu::GpuIndexFlatL2 index_flat;
    
    public:

	kNN::kNN() {}
	kNN::~kNN() {}

        void kNN::fit(float *input, int N, int D) {
            
	    faiss::gpu::StandardGpuResources res;
            this->index_flat(&res, D);
	    index_flat.add(N, input);
        }


        void kNN::search(float *search_items, int search_items_size, long *res_I, float *res_D, int k) {
	    
            index_flat.search(search_items_size, search_items, k, res_D, res_I);
        }



/** @} */

}
;
// end namespace ML
