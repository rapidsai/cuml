
using namespace ML;
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>
#include "utils.h"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#pragma once


//
__global__
void computePijKernel(	volatile float * __restrict__ Pij,
                      	const float * __restrict__ distances,
                      	const float * __restrict__ betas,
                      	const int n,
                      	const int n_neighbors,
                      	const int SIZE)
{
    const int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= SIZE) return;

    const int i = TID / n_neighbors;
    const int j = TID % n_neighbors;
    const float dist = distances[TID];

    if (j == 0 && dist == 0f)
    	Pij[TID] = 0f;
    else
    	Pij[TID] = MLCommon::myExp(-betas[i] * dist);
}



//
__global__
void perplexitySearchKernel(volatile float * __restrict__ betas,
                            volatile float * __restrict__ lower_bound,
                            volatile float * __restrict__ upper_bound,
                            volatile bool * __restrict__ found,
                            const float * __restrict__ neg_entropy,
                            const float * __restrict__ row_sumP,
                            const float log_perplexity_target,
                            const float epsilon,
                            const int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    const float neg_ent = neg_entropy[i];
   	const float sum_P = row_sumP[i];
    float beta = betas[i];

    float min_beta = lower_bound[i];
    float max_beta = upper_bound[i];

    const float perplexity = (neg_ent / sum_P) + MLCommon::myLog(sum_P);
    const float perplexity_diff = perplexity - log_perplexity_target;
    const bool is_found = (perplexity_diff < epsilon && - perplexity_diff < epsilon);


    if (!is_found) {
        if (perplexity_diff > 0) {
            min_beta = beta;
            if (max_beta == FLT_MAX || max_beta == -FLT_MAX)
            	beta *= 2.0f;
            else
            	beta = (beta + max_beta) * 0.5f;	// Mult faster than div
        }
        else {
            max_beta = beta;
            if (min_beta == -FLT_MAX || min_beta == FLT_MAX)
            	beta *= 0.5f;
            else
            	beta = (beta + min_beta) * 0.5f;	// Mult faster than div
        }
        lower_bound[i] = min_beta;
        upper_bound[i] = max_beta;
        betas[i] = beta;
    }
    found[i] = is_found;
}



#include "stats/sum.h"
//
void searchPerplexity(	const float * __restrict__ Pij,
						const float * __restrict__ distances,
						const float perplexity_target,
						const float epsilon,
						const int n,
						const int n_neighbors,
						const int SIZE,
						cudaStream_t stream)
{	
	assert(perplexity_target > 0);
	const float log_perplexity_target = MLCommon::myLog(perplexity_target);

	// Allocate memory
	float *betas;		cuda_calloc(betas, n, 1.0f);
	float *lower_bound; cuda_calloc(lower_bound, n, -FLT_MAX);
	float *upper_bound; cuda_calloc(upper_bound, n, FLT_MAX);
	float *entropy;		cuda_malloc(entropy, SIZE);
	bool *found;		cuda_malloc(found, n);

	// Work out blocksizes
    const int BlockSize1 = 1024;
    const int NBlocks1 = round_up(SIZE, BlockSize1);

    const int BlockSize2 = 128;
    const int NBlocks2 = round_up(n, BlockSize2);


    bool all_found = 0;
    int iter = 0;
    float *row_sumP;	cuda_malloc(row_sumP, n);
    float *neg_entropy;	cuda_malloc(neg_entropy, n);


    // To find minimum in found array
    // From UMAP
    thrust::device_ptr<const bool> found_begin = thrust::device_pointer_cast(found);
    thrust::device_ptr<const bool> found_end = found_begin + n;


    // Find best kernel bandwidth for each row
    while (!all_found && iter < 200) {

    	// Get Gaussian Kernel for each row
    	computePijKernel<<<NBlocks1, BlockSize1>>>(Pij, distances, betas, n, n_neighbors, SIZE);
    	CUDA_CHECK(cudaDeviceSynchronize());

    	// Get entropy for each row
    	// TODO check if ROWSUM works by swapping col,row to row,col
    	sum(row_sumP, Pij, n, n_neighbors, false, stream);


    	// If val = x*log(x) == inf return 0 else val
    	LinAlg::unaryOp(entropy, Pij, SIZE,
						[] __device__(Type x) {
							const float val = x * MLCommon::myLog(x);
 							return (val != val || isinf(val)) ? 0 : val;
						}, 
						stream);


    	// -1 * Row sum(entroy)
    	sum(neg_entropy, entropy, n, n_neighbors, false, stream);
    	LinAlg::unaryOp(neg_entropy, neg_entropy, n,
						[] __device__(Type x) { return -x; }, stream);


    	// Search Perplexity
    	perplexitySearchKernel<<<NBlocks2, BlockSize2>>>(
    		betas, lower_bound, upper_bound,
    		found, neg_entropy, row_sumP, log_perplexity_target, epsilon, n);
    	CUDA_CHECK(cudaDeviceSynchronize());


    	// Check if searching has been completed
    	all_found = *(thrust::min_element(thrust::cuda::par.on(stream), found_start, found_end));


    	iter++;
    }

    cuda_free(neg_entropy);


    // Divide Pij by row_sumP row wise
	LinAlg::matrixVectorOp(Pij, Pij, row_sumP, n_neighbors, n, false, 0,
							[] __device__(Type mat, Type b) { return mat / b; }, stream);


    // Free all allocations
    cuda_free(row_sumP);
    cuda_free(betas);
    cuda_free(lower_bound);
    cuda_free(upper_bound);
    cuda_free(entropy);
    cuda_free(found);
}

