
#pragma once
#include "utils.h"
#include <float.h>
#include <limits.h>


//
__global__
void computePijKernel(  volatile float * __restrict__ Pij,
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

    if (j == 0 && dist == 0.0f)
        Pij[TID] = 0.0f;
    else
        Pij[TID] = EXP(-betas[i] * dist);
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

    const float perplexity = (neg_ent / sum_P) + LOG(sum_P);
    const float perplexity_diff = perplexity - log_perplexity_target;
    const bool is_found = ((perplexity_diff < epsilon) && (-perplexity_diff < epsilon));


    if (!is_found) {
        if (perplexity_diff > 0) {
            min_beta = beta;
            if (max_beta == FLT_MAX || max_beta == -FLT_MAX)
                beta *= 2.0f;
            else
                beta = (beta + max_beta) * 0.5f;    // Mult faster than div
        }
        else {
            max_beta = beta;
            if (min_beta == -FLT_MAX || min_beta == FLT_MAX)
                beta *= 0.5f;
            else
                beta = (beta + min_beta) * 0.5f;    // Mult faster than div
        }
        lower_bound[i] = min_beta;
        upper_bound[i] = max_beta;
        betas[i] = beta;
    }
    found[i] = is_found;
}



// If val = x*log(x) == inf return 0 else val
// Let function be compiled first so access in future faster
static void xlogx(float *entropy, float *Pij, int SIZE, cudaStream_t stream) {
    LinAlg::unaryOp(entropy, Pij, SIZE,
                    [] __device__(float x) {
                        if (x <= 0.0f || isinf(x)) return 0.0f;
                        return x * LOG(x);
                        // since log(x) bounded approx -50, 50.
                    }, 
                    stream);
}



// Searches for the best perplexity and stores it
// in Pij
namespace Perplexity_Search_ {

void searchPerplexity(  float * __restrict__ Pij,
                        const float * __restrict__ distances,
                        const float perplexity_target, // desired perplexity
                        const float epsilon,
                        const int n,
                        const int n_neighbors,
                        const int SIZE,
                        cudaStream_t stream)
{   
    assert(perplexity_target > 0);
    const float log_perplexity_target = LOG(perplexity_target);

    // Allocate memory
    float *betas;       cuda_calloc(betas, n, 1.0f);
    float *lower_bound; cuda_calloc(lower_bound, n, 0.0f);
    float *upper_bound; cuda_calloc(upper_bound, n, 1000.0f);
    float *entropy;     cuda_malloc(entropy, SIZE);
    bool *found;        cuda_malloc(found, n);

    // Work out blocksizes
    const int BlockSize1 = 1024;
    const int NBlocks1 = Utils_::ceildiv(SIZE, BlockSize1);

    const int BlockSize2 = 128;
    const int NBlocks2 = Utils_::ceildiv(n, BlockSize2);


    bool all_found = 0;
    float *row_sumP;    cuda_malloc(row_sumP, n);
    float *neg_entropy; cuda_malloc(neg_entropy, n);


    // To find minimum in found array
    // From UMAP
    thrust::device_ptr<bool> found_begin = thrust::device_pointer_cast(found);
    thrust::device_ptr<bool> found_end = found_begin + n;


    // Find best kernel bandwidth for each row
    int iter = 0;
    while (!all_found && iter < 200) {

        // Get Gaussian Kernel for each row
        computePijKernel<<<NBlocks1, BlockSize1>>>(
            Pij, distances, betas, n, n_neighbors, SIZE);
        cuda_synchronize();

        // Get entropy for each row
        // TODO check if ROWSUM works by swapping col,row to row,col
        Utils_::row_sum(row_sumP, Pij, n, n_neighbors, stream);


        // If val = x*log(x) == inf return 0 else val
        xlogx(entropy, Pij, SIZE, stream);


        // -1 * Row_sum(entropy)
        Utils_::row_sum(neg_entropy, entropy, n, n_neighbors, stream);
        LinAlg::scalarMultiply(neg_entropy, neg_entropy, -1.0f, n, stream);


        // Search Perplexity
        perplexitySearchKernel<<<NBlocks2, BlockSize2>>>(
            betas, lower_bound, upper_bound,
            found, neg_entropy, row_sumP, log_perplexity_target, epsilon, n);
        cuda_synchronize();


        // Check if searching has been completed
        all_found = Utils_::min_array(found_begin, found_end, stream);

        iter++;
    }


    // Free all allocations
    cuda_free(neg_entropy);
    cuda_free(betas);
    cuda_free(lower_bound);
    cuda_free(upper_bound);
    cuda_free(entropy);
    cuda_free(found);


    // Pij / row_sumP.reshape(-1,1)
    // NOTICE since division slower, perform 1/row_sumP first
    LinAlg::unaryOp(row_sumP, row_sumP, n, 
                    [] __device__(float x) {
                        if (x == 0.0f) return 0.0f;
                        return 1.0f/x;
                    },
                    stream);

    LinAlg::matrixVectorOp(Pij, Pij, row_sumP, n_neighbors, n, false, 0,
                            [] __device__(float mat, float b) { return mat*b; },
                            stream);
    // Last free
    cuda_free(row_sumP);
}



using namespace MLCommon::Sparse;
// Symmetrize P + P.T after perplexity searching
void P_add_PT(  const long * __restrict__ indices,
                const int n,
                const int n_neighbors,
                COO<float> * __restrict__ P,
                COO<float> * __restrict__ P_PT,
                cudaStream_t stream)
{
    // Perform P + P.T
    // Notice doubles memory allocation
    MLCommon::Sparse::coo_symmetrize(P, P_PT,
        [] __device__ (int row, int col, float val, float trans) {
            return val + trans;
        }, stream);
    cuda_synchronize();
}


// end namespace
}

