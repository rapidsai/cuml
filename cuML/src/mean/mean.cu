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

#pragma once

#include "mean_c.h"
#include <stats/mean.h>
#include <iostream>
#include <cuda_runtime.h>

namespace ML {

using namespace MLCommon;

void mean(float* mu, float* data, int D, int N, bool sample, bool rowMajor) {
        //std::cout << "C++ Allocated output array: "  << static_cast<void*>(mu) << std::endl;
	//std::cout << "C++ N: " << N << std::endl;
	//std::cout << "C++ D: " << D << std::endl;
	//std::cout << "C++ Allocated input array: " << static_cast<void*>(data) << std::endl;
	//std::cout << "Setting device..." << std::endl;

            cudaPointerAttributes att;
            cudaError_t err = cudaPointerGetAttributes(&att, data);

            std::cout << "ERR: " << err << std::endl
             << "MEMORY TYPE: " << att.memoryType << std::endl;


        Stats::mean(mu, data, D, N, sample, rowMajor);
}

void mean(double* mu, double* data, int D, int N, bool sample, bool rowMajor) {

	Stats::mean(mu, data, D, N, sample, rowMajor);
}

/** @} */

}
;
// end namespace ML
