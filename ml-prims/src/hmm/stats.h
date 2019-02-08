#include <random/rng.h>
#include <linalg/cublas_wrappers.h>
#include <ml_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#define IDX2C(i,j,ld) (j*ld + i)

using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {



template <typename T>
void weighted_mean(T* weights, T* data, int dim, int n_samples){
        // Write functor and apply
}

template <typename T>
void weighted_cov(T* weights, T* data, int dim, int n_samples){
        // Write functor and apply
}

}
}
