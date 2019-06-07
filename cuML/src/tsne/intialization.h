
// From UMAP / init_embed
#pragma once
#include "utils.h"
#include "random/rng.h"
#include "sys/time.h"

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;


__global__ void InitializationKernel(int * __restrict errd) {
    *errd = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}


//
namespace Intialization_ {


int *Initialize(void) {
    // Intialize cache levels and errors
    int *errd;       cuda_malloc(errd, 1);
    //
    InitializationKernel<<<1, 1>>>(errd);
    cuda_synchronize();

    // Get GPU information
    cudaDeviceProp GPU_info;
    cudaGetDeviceProperties(&GPU_info, 0);

    if (GPU_info.warpSize != WARPSIZE) {
        fprintf(stderr, "Warp size must be %d\n", GPU_info.warpSize);
        cuda_free(errd);
        exit(-1);
    }
    return errd;
}


// Creates a random uniform vector
float *randomVector(const float minimum,
                    const float maximum,
                    const int size, 
                    const long long seed,
                    cudaStream_t stream) {
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