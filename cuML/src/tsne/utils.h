
using namespace ML;
#define TPB_X 32

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <iostream>
#include <math.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define cuda_free(x) 	CUDA_CHECK(cudaFree(x))
#define exp(x)			MLCommon::myExp(x)
#define log(x)			MLCommon::myLog(x)
#define MAX(a, b)		((a > b) ? a : b)
#define MIN(a, b)		((a < b) ? a : b)



// Malloc some space
template <typename Type>
void cuda_malloc(Type *&ptr, const size_t n) {
	// From ml-prims / src / utils.h
	CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
}


// Malloc and memset some space
template <typename Type>
void cuda_calloc(Type *&ptr, const size_t n, const Type val) {
	// From ml-prims / src / utils.h
	// Just allows easier memsetting
	CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
	CUDA_CHECK(cudaMemset(ptr, val, sizeof(Type) * n));
}


// Determines number of blocks
// Similar to page size determination --> need to round up
inline int ceildiv(const int a, const int b) {
	if (a % b != 0)
		return a/b + 1;
	return a/b;
}


namespace Utils_ {

// Finds minimum(array) from UMAP
template <typename Type>
inline Type min_array(	thrust::device_ptr<const Type> start, 
						thrust::device_ptr<const Type> end,
						cudaStream_t stream) 
{
	return *(thrust::min_element(thrust::cuda::par.on(stream), start, end))
}


// Finds maximum(array) from UMAP
template <typename Type>
inline Type max_array(	thrust::device_ptr<const Type> start, 
						thrust::device_ptr<const Type> end,
						cudaStream_t stream) 
{
	return *(thrust::max_element(thrust::cuda::par.on(stream), start, end))
}



#include "stats/sum.h"
// Does row_sum on C contiguous data
inline void row_sum(float *out, const float *in, 
					const int n, const int p,
					cudaStream_t stream)
{
	// Since sum is for F-Contiguous columns, then C-contiguous rows
	// is also fast since it is transposed.
	//				    dim, rows, rowMajor
	Stats::sum(out, in,  n,   p,    false, stream);
}


} // end namespace
