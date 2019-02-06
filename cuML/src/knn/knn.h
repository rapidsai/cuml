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

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/IndexProxy.h>
#include <iostream>
#include <faiss/gpu/GpuIndexFlat.h>

#ifndef _KNN_H
#define _KNN_H
namespace ML {


    class kNN {


 	   std::vector<faiss::gpu::StandardGpuResources* > res;
 	   std::vector<faiss::gpu::GpuIndexFlatL2* > sub_indices;


    	faiss::gpu::IndexProxy indexProxy;
     	int D;

    public:
			kNN(int D);
			~kNN();
			void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k);
			void fit(float *input, int N, int n_gpus);

    };
}

#endif
