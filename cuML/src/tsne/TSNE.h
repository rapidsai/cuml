
using namespace ML;

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>
#define cuda_free(x)        CUDA_CHECK(cudaFree(x))
#define cuda_malloc(x, n)   MLCommon::allocate(x, n);


template <typename Type>
void cuda_calloc(Type *&ptr, size_t n, Type val) {
    // From ml-prims / src / utils.h
    // Just allows easier memsetting
    CUDA_CHECK(cudaMalloc((void **)&ptr, sizeof(Type) * n));
    CUDA_CHECK(cudaMemset(ptr, val, sizeof(Type) * n));
}


__global__ void upper_lower_assign( float * __restrict__ sigmas,
                                    float * __restrict__ lower_bound,
                                    float * __restrict__ upper_bound,
                                    const float * __restrict__ perplexity,
                                    const float target_perplexity,
                                    const int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        if (perplexity[i] > target_perplexity)
            upper_bound[i] = sigmas[i];
        else
            lower_bound[i] = sigmas[i];
        sigmas[i] = (upper_bound[i] + lower_bound[i]) * 0.5f;
    }
}


#include "distance/euclidean.h"
#include "linalg/power.h"
#include "linalg/matrix_vector_op.h"
template<typename Type>
void TSNE::compute_Pij( float * __restrict__ Pij;
                        const Type * __restrict__ X,
                        float * __restrict__ sigmas;
                        const int n,
                        const int p,
                        cudaStream_t * stream,
                        const int full_size,
                        float * __restrict__ sigma_squared)
{   
    // Workspace needed to get rowNorm(X)
    size_t lwork = n * sizeof(Type);
    float *work; cuda_malloc(work, n);

    // From distances / euclidean algo1
                        // row  col col  in  in  out  sqrt   work  lwork  norm  stream
    Distance::euclideanAlgo1(n,  p,  p,  X,  X,  Pij, false, work, lwork,   2,  stream);

    float *sigma_squared; cuda_malloc(sigma_squared, n);
    // From linalg / power
    LinAlg::powerScalar(sigma_squared, sigmas, 2.0f, n, stream);

    // Pij /= sigmas column_size
    // From stats/mean_centre --> but inplace
                        // out   in     divisor    col row  C-cont axis
    LinAlg::matrixVectorOp(Pij, Pij, sigma_squared, p,  n,  false,  0,
                            [] __device__(Type mat, Type b) { return mat / b; },
                            stream);

    // Exp(Pij)
    // From matrix/math.h power and linalg/sqrt
    LinAlg::unaryOp(Pij, Pij, full_size,
                            [] __device__(Type val) { return MLCommon::myExp(val) }, 
                            stream);

    // Diagonal(Pij) = 0
    

    cuda_free(sigma_squared);
    cuda_free(work);
}



template<typename Type>
void TSNE::search_perplexity(   Type * __restrict__ X,
                                const float target_perplexity,
                                const float eps,
                                const int n,
                                const int p)
{
    const int full_size = n*n;
    assert(full_size > 0) // Check overflow

    float *sigmas;      cuda_calloc(sigmas, n, 500.0f);
    float *best_sigmas; cuda_malloc(best_sigmas, n);
    float *perplexity;  cuda_calloc(perplexity, n);
    float *lower_bound; cuda_calloc(lower_bound, 0.0f);
    float *upper_bound; cuda_calloc(upper_bound, 1000.0f);


    float *Pij; cuda_malloc(Pij, full_size);

    // From glm.cu
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    //

    TSNE::compute_Pij(Pij, X, sigmas, n, p, &stream, full_size);

    CUDA_CHECK(cudaStreamDestroy(stream));


    cuda_free(Pij);

    cuda_free(upper_bound);
    cuda_free(lower_bound);
    cuda_free(perplexity);
    cuda_free(best_sigmas);
    cuda_free(sigmas);
}