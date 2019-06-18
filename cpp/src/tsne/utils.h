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

#define TEST_NNZ 12021

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_utils.h>

/* Change to <sparse/coo.h> */
#include "./sparse/coo.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
using namespace thrust::placeholders;

#include <random/rng.h>
#include <stats/sum.h>
#include <sys/time.h>

#define MIN(a, b) ((a > b) ? b : a)
#define MAC(a, b) ((a < b) ? b : a)


namespace ML {
namespace TSNE {

void
random_vector(float *vector, const float minimum, const float maximum,
                     const int size, cudaStream_t stream, long long seed = -1) {
    if (seed <= 0) {
        // Get random seed based on time of day
        struct timeval tp;
        gettimeofday(&tp, NULL);
        seed = tp.tv_sec * 1000 + tp.tv_usec;
    }
    MLCommon::Random::Rng random(seed);
    random.uniform<float>(vector, size, minimum, maximum, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
}  // namespace ML
