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

#include <curand.h>


namespace MLCommon {
namespace Random {

/** check for curand runtime API errors and assert accordingly */
#define CURAND_CHECK(call)                              \
    do {                                                \
        curandStatus_t status = call;                   \
        ASSERT(status == CURAND_STATUS_SUCCESS,         \
               "FAIL: curand-call='%s'. Reason:%d\n",   \
               #call, status);                          \
    } while(0)

/**
 * @defgroup normal curand normal random number generation operations
 * @{
 */
template <typename T>
curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                    T *outputPtr, size_t n, T mean, T stddev);

template <>
inline curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                           float *outputPtr, size_t n,
                                           float mean, float stddev) {
    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

template <>
inline curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                           double *outputPtr, size_t n,
                                           double mean, double stddev) {
    return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
/** @} */

}; // end namespace Random
}; // end namespace MLCommon
