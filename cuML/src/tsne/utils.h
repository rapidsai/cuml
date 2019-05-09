
using namespace ML;
#define TPB_X 32

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>

#define cuda_free(x) CUDA_CHECK(cudaFree(x))


template <typename Type>
void cuda_malloc(Type *&ptr, const size_t n) {
	// From ml-prims / src / utils.h
	CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
}


template <typename Type>
void cuda_calloc(Type *&ptr, const size_t n, const Type val) {
	// From ml-prims / src / utils.h
	// Just allows easier memsetting
	CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
	CUDA_CHECK(cudaMemset(ptr, val, sizeof(Type) * n));
}


inline int round_up(const int a, const int b) {
	if (a % b != 0)
		return a/b + 1;
	return a/b;
}

