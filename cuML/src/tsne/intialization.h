
// From UMAP / init_embed

using namespace ML;
#include "utils.h"
#include "random/rng.h"
#include "sys/time.h"

#pragma once


__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;

__global__ void InitializationKernel(int * __restrict err)
{
    *err = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}


//
namespace Intialization_ {


void Initialize(int *err) 
{
	// TODO set cache levels
    InitializationKernel<<<1, 1>>>(err);
    cuda_synchronize();
}



float *randomVector(const float minimum,
					const float maximum,
					const int size, 
					const long long seed,
					cudaStream_t stream)
{
	if (seed < 0) {
		struct timeval tp;
	    gettimeofday(&tp, NULL);
	    seed = tp.tv_sec * 1000 + tp.tv_usec;
	}

	float *embedding;
    MLCommon::Random::Rng random(seed);
    random.uniform<float>(embedding, size, minimum, maximum, stream);
    return embedding;
}

// end namespace
}