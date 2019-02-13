#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "utils.h"
#include "cuda_utils.h"
#include "linalg/eltwise.h"
#include "linalg/transpose.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"

#include <hmm/utils.h>


using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {




template <typename T>
struct StrideFunctor
{
        int stride;
        T* array;
        StrideFunctor(int _stride, T* _array){
                stride = _stride;
                array = _array;
        }

        __host__ __device__
        T operator() (int idx)
        {
                return *(array + idx * stride);
        }
};

template <typename T>
T diag_product(T* array, int nDim){
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + nDim;
        StrideFunctor<T> strideFn(nDim + 1, array);
        return thrust::reduce(thrust::make_transform_iterator(first, strideFn),
                              thrust::make_transform_iterator(last, strideFn),
                              (T) 1., thrust::multiplies<T>());
}


template <typename T>
struct Determinant
{
        int nDim;
        T *tempM;

        // Cusolver variables
        int *info, info_h;
        cusolverDnHandle_t cusolverHandle;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int cholWsSize;
        T *cholWs = nullptr;

        Determinant(int _nDim){
                nDim = _nDim;
                CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
                CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(cusolverHandle, uplo, nDim, tempM, nDim, &cholWsSize));
                allocate(cholWs, cholWsSize);
                allocate(info, 1);
                allocate(tempM, nDim * nDim);

        }

        // We assume M is hermitian !!!
        T compute(T* M){
                // TODO : Add Hermitian assertion
                // Copy to tempM
                copy(tempM, M, nDim * nDim);

                // Compute Cholesky decomposition
                // TODO : Optimize memory usage
                CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(cusolverHandle, uplo,
                                                       nDim, tempM, nDim,
                                                       cholWs, cholWsSize, info));
                updateHost(&info_h, info, 1);

                ASSERT(info_h == 0, "sigma: error in potrf, info=%d | expected=0", info_h);

                // Compute determinant
                T prod =  diag_product(tempM, nDim);
                return prod * prod;
        }
};


}
}
