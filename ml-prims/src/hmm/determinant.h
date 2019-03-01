#pragma once

#include <cublas_v2.h>

#include "utils.h"
#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"


namespace MLCommon {

template <typename T>
struct Determinant
{
        int nDim, lda;
        T *tempM;
        bool is_hermitian;

        int *devIpiv;

        int *info, info_h;
        cusolverDnHandle_t *cusolverHandle;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int cholWsSize;
        T *cholWs = NULL;

        Determinant(int _nDim, int _lda, cusolverDnHandle_t *_cusolverHandle,
                    bool _is_hermitian){
                nDim = _nDim;
                lda = _lda;
                is_hermitian = _is_hermitian;
                cusolverHandle = _cusolverHandle;
                CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(*cusolverHandle, uplo, nDim, tempM, nDim, &cholWsSize));

                allocate(cholWs, cholWsSize);
                allocate(info, 1);
                allocate(tempM, nDim * nDim);
                if (is_hermitian) {
                        allocate(devIpiv, nDim);
                }
        }

        T compute(T* M){
                copy(tempM, M, nDim * nDim);

                if(is_hermitian) {
                        CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(*cusolverHandle,
                                                               uplo,
                                                               nDim,
                                                               tempM,
                                                               nDim,
                                                               cholWs,
                                                               cholWsSize,
                                                               info));
                }
                else{
                        CUSOLVER_CHECK(LinAlg::cusolverDngetrf(*cusolverHandle,
                                                               _nDim,
                                                               _nDim,
                                                               tempM,
                                                               lda,
                                                               cholWs,
                                                               devIpiv,
                                                               info));

                }
                updateHost(&info_h, info, 1);
                ASSERT(info_h == 0,
                       "sigma: error in potrf, info=%d | expected=0", info_h);

                T prod = diag_product(tempM, nDim);
                return prod * prod;
        }

        void TearDown() {
                CUDA_CHECK(cudaFree(cholWs));
                CUDA_CHECK(cudaFree(tempM));
                CUDA_CHECK(cudaFree(info));

                if (is_hermitian) {
                        CUDA_CHECK(cudaFree(devIpiv));
                }
        }

};

}
