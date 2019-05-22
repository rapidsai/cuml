
// From UMAP / init_embed

using namespace ML;
#include "utils.h"
#include "random/rng.h"
#include "sys/time.h"

#pragma once

#define set_cache(a, b) cudaFuncSetCacheConfig(a, b)


__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;

__global__ void InitializationKernel(int * __restrict errd)
{
    *errd = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}


//
namespace Intialization_ {


void Initialize(int *errd) 
{
    // TODO set cache levels
    set_cache(BoundingBox_::boundingBoxKernel, cudaFuncCachePreferShared);

    //
    InitializationKernel<<<1, 1>>>(errd);
    cuda_synchronize();
}



float *randomVector(const float minimum,
                    const float maximum,
                    const int size, 
                    const long long seed,
                    cudaStream_t stream)
{
    // from UMAP
    if (seed < 0) {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        seed = tp.tv_sec * 1000 + tp.tv_usec;
    }

    float *vector;
    MLCommon::Random::Rng random(seed);
    random.uniform<float>(vector, size, minimum, maximum, stream);
    return vector;
}

// end namespace
}