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
