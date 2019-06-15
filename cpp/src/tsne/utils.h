
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


// ###################### General global funcs ######################
#define thrust_t thrust::device_ptr
#define to_thrust thrust::device_pointer_cast
#define __STREAM__ thrust::cuda::par.on(stream)
#define cuda_create_stream(x) CUDA_CHECK(cudaStreamCreate(x))
#define cuda_destroy_stream(x) CUDA_CHECK(cudaStreamDestroy(x))
#define cuda_synchronize() CUDA_CHECK(cudaDeviceSynchronize())

#define COO_t Sparse::COO
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a > b) ? b : a)


// ###################### Functions ######################
namespace ML {
using namespace ML;
using namespace MLCommon;


// ###################### Debugging prints ######################
#define DEBUG(fmt, ...)                                \
	do {                                                 \
		if (IF_DEBUG) fprintf(stderr, fmt, ##__VA_ARGS__); \
	} while (0);


// ###################### Creates a random uniform vector ######################
void random_vector(float *vector, const float minimum, const float maximum,
					const int size, cudaStream_t stream, long long seed = -1) {
	if (seed <= 0) {
		// Get random seed based on time of day
		struct timeval tp;
		gettimeofday(&tp, NULL);
		seed = tp.tv_sec * 1000 + tp.tv_usec;
	}
	Random::Rng random(seed);
	random.uniform<float>(vector, size, minimum, maximum, stream);
	CUDA_CHECK(cudaPeekAtLastError());
}


inline void array_multiply(float *array, const float mult, const int n, cudaStream_t stream) {
	thrust_t<float> begin = to_thrust(array);
	thrust::transform(__STREAM__, begin, begin + n, begin, mult * _1);
}


}
